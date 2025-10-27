#include "conv.h"
#include "file.h"

float* alloc_aligned(size_t count) {
    if (!count) return NULL;
    void* ptr = NULL;
    if (posix_memalign(&ptr, ALIGN_BYTES, count * sizeof(float)) != 0) return NULL;
    return (float*)ptr;
}

void calc_output_dims(ConvParams* params) {
    params->out_H = (uint32_t)calc_output_height((int)params->H, (int)params->kH, (int)params->sH);
    params->out_W = (uint32_t)calc_output_width((int)params->W, (int)params->kW, (int)params->sW);
}

void calc_input_rows_for_output_range_clamped(uint32_t out_row_start,
                                              uint32_t out_row_end,
                                              uint32_t sH,
                                              uint32_t kH,
                                              uint32_t max_input_H,
                                              uint32_t* input_row_start,
                                              uint32_t* num_input_rows) {
    int half = (int)(kH - 1) / 2;
    int start = (int)(out_row_start * sH);
    int end = (int)((out_row_end ? out_row_end - 1 : out_row_start) * sH);

    int in_start = start - half;
    int in_end = end + (int)kH - half;

    if (in_start < 0) in_start = 0;
    if (in_end > (int)max_input_H) in_end = (int)max_input_H;

    *input_row_start = (uint32_t)in_start;
    *num_input_rows = (uint32_t)((in_end > in_start) ? (in_end - in_start) : 0);
}

uint32_t calc_chunk_size(uint32_t W,
                         uint32_t out_W,
                         uint32_t kH,
                         uint32_t kW,
                         uint32_t sH,
                         size_t budget_bytes) {
    uint32_t rows_per_out = sH + kH;
    size_t kernel_bytes = (size_t)kH * kW * sizeof(float);
    size_t margin = budget_bytes > kernel_bytes ? budget_bytes - kernel_bytes : budget_bytes / 2;
    if (!margin) margin = budget_bytes;
    size_t row_bytes = (size_t)(rows_per_out * W + out_W) * sizeof(float);
    if (!row_bytes) return 1;
    uint32_t chunk = (uint32_t)(margin / row_bytes);
    return chunk ? chunk : 1;
}

ConvParams* init_conv_params(const char* input_file,
                             const char* kernel_file,
                             uint32_t sH,
                             uint32_t sW) {
    ConvParams* params = (ConvParams*)malloc(sizeof(ConvParams));
    if (!params) return NULL;

    BinaryFile input = open_bin_matrix_input((char*)input_file);
    BinaryFile kernel = open_bin_matrix_input((char*)kernel_file);
    if (!input.file || !kernel.file) {
        if (input.file) fclose(input.file);
        if (kernel.file) fclose(kernel.file);
        free(params);
        return NULL;
    }

    params->H = input.height;
    params->W = input.width;
    params->kH = kernel.height;
    params->kW = kernel.width;
    params->sH = sH;
    params->sW = sW;
    calc_output_dims(params);

    size_t input_elems = (size_t)params->H * (size_t)params->W;
    size_t kernel_elems = (size_t)params->kH * (size_t)params->kW;
    size_t output_elems = (size_t)params->out_H * (size_t)params->out_W;

    params->data = alloc_aligned(input_elems);
    params->kernel = alloc_aligned(kernel_elems);
    params->output = alloc_aligned(output_elems);

    if (!params->data || !params->kernel || !params->output) {
        fclose(input.file);
        fclose(kernel.file);
        free(params->data);
        free(params->kernel);
        free(params->output);
        free(params);
        return NULL;
    }

    fread(params->data, sizeof(float), input_elems, input.file);
    fread(params->kernel, sizeof(float), kernel_elems, kernel.file);

    fclose(input.file);
    fclose(kernel.file);
    return params;
}

void free_conv_params(ConvParams* params) {
    if (!params) return;
    free(params->data);
    free(params->kernel);
    free(params->output);
    free(params);
}
