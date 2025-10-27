#pragma once

#include <stdint.h>
#include <mpi.h>

typedef struct {
    int global_h;
    int global_w;
    int start_row;
    int start_col;
    int block_h;
    int block_w;
} Subarray2D;

int mpi_read_subarray_f32(const char* filepath,
                          const Subarray2D* sub,
                          float* recv_buffer,
                          MPI_Comm comm);

int mpi_write_subarray_f32(const char* filepath,
                           int out_global_h,
                           int out_global_w,
                           const Subarray2D* sub,
                           const float* send_buffer,
                           MPI_Comm comm);


