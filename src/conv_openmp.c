#include "conv.h"
#include <omp.h>

static inline float apply_window(const float* __restrict__ input_data,
                                 const float* __restrict__ kernel_data,
                                 uint32_t H, uint32_t W,
                                 uint32_t center_row, uint32_t center_col,
                                 uint32_t kH, uint32_t kW) {
    float sum = 0.0f;
    
    int half_h = (int)(kH - 1) / 2;
    int half_w = (int)(kW - 1) / 2;
    
    for (uint32_t k_i = 0; k_i < kH; k_i++) {
        for (uint32_t k_j = 0; k_j < kW; k_j++) {
            int i = (int)center_row + (int)k_i - half_h;
            int j = (int)center_col + (int)k_j - half_w;
            
            float sample = 0.0f;
            if (i >= 0 && i < (int)H && j >= 0 && j < (int)W) {
                sample = input_data[i * W + j];
            }
            
            sum += sample * kernel_data[k_i * kW + k_j];
        }
    }
    
    return sum;
}

void conv_openmp(ConvParams* params) {
    const uint32_t H = params->H;
    const uint32_t W = params->W;
    const uint32_t kH = params->kH;
    const uint32_t kW = params->kW;
    const uint32_t sH = params->sH;
    const uint32_t sW = params->sW;
    const uint32_t out_H = params->out_H;
    const uint32_t out_W = params->out_W;
    const uint32_t input_offset = params->input_offset_row;
    const uint32_t output_offset = params->output_offset_row;
    
    const float* kernel = params->kernel;

    #pragma omp parallel
    {
        float* local_kernel = (float*)malloc(kH * kW * sizeof(float));
        if (local_kernel) {
            memcpy(local_kernel, params->kernel, kH * kW * sizeof(float));
        }

        #pragma omp for schedule(static)
        for (uint32_t out_row = 0; out_row < out_H; ++out_row) {
            const uint32_t row_center = (out_row + output_offset) * sH - input_offset;
            float* output_row = params->output + out_row * out_W;
            float* kernel_data = local_kernel ? local_kernel : params->kernel;
            uint32_t col_center = 0;

            for (uint32_t out_col = 0; out_col < out_W; ++out_col) {
                float value = apply_window(
                    params->data, kernel_data,
                    H, W, row_center, col_center, kH, kW
                );

                output_row[out_col] = value;
                col_center += sW;
            }
        }

        if (local_kernel) {
            free(local_kernel);
        }
    }
}
