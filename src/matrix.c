#include "matrix.h"
#include <stdlib.h>

MatrixPadding dim_to_padding(uint32_t kH, uint32_t kW) {
    MatrixPadding pad = (MatrixPadding){0, 0, 0, 0};
    if (!kH || !kW) return pad;

    pad.pad_h_b = (kH - 1) / 2;
    pad.pad_h_a = kH / 2;
    pad.pad_w_b = (kW - 1) / 2;
    pad.pad_w_a = kW / 2;

    return pad;
}

int calc_output_height(int H, int kH, int sH) {
    (void)kH;
    return ((H - 1) / sH) + 1;
}

int calc_output_width(int W, int kW, int sW) {
    (void)kW;
    return ((W - 1) / sW) + 1;
}

int calc_steps_total(int H, int W, int kH, int kW, int sH, int sW) {
    int out_H = calc_output_height(H, kH, sH);
    int out_W = calc_output_width(W, kW, sW);
    return out_H * out_W;
}

int calc_steps_dim(int X, int kX, int sX) {
    return calc_output_height(X, kX, sX);
}
