#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

typedef struct {
    uint32_t height;
    uint32_t width;
    void* v;
} MatrixHeader;

typedef struct {
    uint16_t pad_h_b;
    uint16_t pad_h_a;

    uint16_t pad_w_b;
    uint16_t pad_w_a;
} MatrixPadding;

MatrixPadding dim_to_padding(uint32_t kH, uint32_t kW);

int calc_output_height(int H, int kH, int sH);
int calc_output_width(int W, int kW, int sW);
int calc_steps_total(int H, int W, int kH, int kW, int sH, int sW);
int calc_steps_dim(int X, int kX, int sX);

#endif
