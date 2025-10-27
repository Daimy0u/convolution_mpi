#ifndef CONV_H
#define CONV_H

#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>
#include "matrix.h"

#define ALIGN_BYTES 64

// convolution parameters
typedef struct {
    float* data;        // input chunk
    float* kernel;      // kernel
    float* output;      // output chunk
    uint32_t H, W;      // chunk dims
    uint32_t kH, kW;    // kernel dims
    uint32_t sH, sW;    // stride
    uint32_t out_H;     // output dims
    uint32_t out_W;
    uint32_t input_offset_row;   // global input row offset
    uint32_t output_offset_row;  // global output row offset
} ConvParams;

void conv_openmp(ConvParams *params);
void conv_mpi(ConvParams *params, MPI_Comm comm, const char *input_path, const char *output_path, size_t budget_bytes);

float* alloc_aligned(size_t n);
void calc_output_dims(ConvParams* params);
void calc_input_rows_for_output_range_clamped(uint32_t out_row_start,
                                              uint32_t out_row_end,
                                              uint32_t sH,
                                              uint32_t kH,
                                              uint32_t max_input_H,
                                              uint32_t* input_row_start,
                                              uint32_t* num_input_rows);
uint32_t calc_chunk_size(uint32_t W,
                         uint32_t out_W,
                         uint32_t kH,
                         uint32_t kW,
                         uint32_t sH,
                         size_t budget_bytes);
ConvParams* init_conv_params(const char* input_file,
                             const char* kernel_file,
                             uint32_t sH,
                             uint32_t sW);
void free_conv_params(ConvParams* params);
#endif
