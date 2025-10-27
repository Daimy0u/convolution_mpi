#ifndef FILE_H
#define FILE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "matrix.h"

typedef struct {
    uint32_t height;
    uint32_t width;
} BinaryHeader;

typedef struct {
    uint32_t height;
    uint32_t width;
    size_t start;
} BinaryOffset;

typedef struct {
    uint32_t height;
    uint32_t width;
    FILE* file;
} BinaryFile;

FILE* create_bin_matrix(char* filepath,  uint32_t h, uint32_t w);
BinaryFile open_bin_matrix_input(char* filepath);

void apply_padding_bin(char* src_fp,char* dst_fp, MatrixPadding* padding, size_t chunk_size);
void apply_imap_bin(char* padded_bin_fp, char* im2col_bin_fp, uint32_t kH, uint32_t kW, uint32_t sH, uint32_t sW, MatrixPadding* padding, size_t chunk_size);

void convert_txt_to_bin(char* txt_fp, char* bin_fp, size_t chunk_size);
void convert_bin_to_txt(char* bin_fp, char* txt_fp, size_t chunk_size);

#endif // FILE_H
