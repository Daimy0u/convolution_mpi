#include "conv.h"
#include "file.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    uint32_t chunk_start;
    uint32_t chunk_end;
    uint32_t chunk_out_H;
    uint32_t input_row_start;
    uint32_t num_input_rows;
    MPI_Offset input_offset;
    MPI_Offset output_offset;
} Chunk;

static void build_chunk(Chunk* chunk,
                        uint32_t chunk_start,
                        uint32_t chunk_rows,
                        uint32_t row_end,
                        uint32_t sH,
                        uint32_t kH,
                        uint32_t input_H,
                        uint32_t W,
                        uint32_t out_W) {
    uint32_t chunk_end = chunk_start + chunk_rows;
    if (chunk_end > row_end) chunk_end = row_end;

    chunk->chunk_start = chunk_start;
    chunk->chunk_end = chunk_end;
    chunk->chunk_out_H = chunk_end - chunk_start;

    calc_input_rows_for_output_range_clamped(chunk_start,
                                             chunk_end,
                                             sH,
                                             kH,
                                             input_H,
                                             &chunk->input_row_start,
                                             &chunk->num_input_rows);

    chunk->input_offset = (MPI_Offset)sizeof(BinaryHeader)
                        + (MPI_Offset)chunk->input_row_start 
                        * (MPI_Offset)W * (MPI_Offset)sizeof(float);
    chunk->output_offset = (MPI_Offset)sizeof(BinaryHeader)
                         + (MPI_Offset)chunk_start * (MPI_Offset)out_W 
                         * (MPI_Offset)sizeof(float);
}

void conv_mpi(ConvParams* params,
              MPI_Comm comm,
              const char* input_path,
              const char* output_path,
              size_t budget_bytes) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const uint32_t W = params->W;
    const uint32_t kH = params->kH;
    const uint32_t kW = params->kW;
    const uint32_t sH = params->sH;
    const uint32_t sW = params->sW;
    const uint32_t out_H = params->out_H;
    const uint32_t out_W = params->out_W;

    size_t rank_budget = budget_bytes / (size_t)size;

    uint32_t rows_per_rank = (out_H + size - 1) / size;
    uint32_t row_start = rank * rows_per_rank;
    uint32_t row_end = row_start + rows_per_rank;
    if (row_end > out_H) row_end = out_H;
    uint32_t row_count = row_end - row_start;

    if (!row_count) return;

    uint32_t chunk_rows = calc_chunk_size(W, out_W, kH, kW, sH, rank_budget);
    uint32_t chunk_total = (row_count + chunk_rows - 1) / chunk_rows;

    if (rank == 0) {
        printf("[MPI] ranks=%d mem_total=%.3fGB mem_per_rank=%.3fGB chunk_rows=%u out_size=%ux%u\n",
               size, budget_bytes / 1e9, rank_budget / 1e9, chunk_rows, out_H, out_W);
    }
    printf("[MPI] rank=%d rows=%u-%u chunks=%u\n", rank, row_start, row_end, chunk_total);

    MPI_File input_file, output_file;
    MPI_Info info_in, info_out;
    MPI_Info_create(&info_in);
    MPI_Info_set(info_in, "romio_cb_read", "enable");
    MPI_Info_set(info_in, "access_style", "read_once,sequential");

    MPI_Info_create(&info_out);
    MPI_Info_set(info_out, "romio_cb_write", "enable");
    MPI_Info_set(info_out, "access_style", "write_once,sequential");

    int mpi_err = MPI_File_open(comm, (char*)input_path, MPI_MODE_RDONLY, info_in, &input_file);
    if (mpi_err != MPI_SUCCESS) {
        char err_string[MPI_MAX_ERROR_STRING];
        int err_len = 0;
        MPI_Error_string(mpi_err, err_string, &err_len);
        fprintf(stderr, "[Rank %d] Failed to open input file '%s': %.*s\n", rank, input_path, err_len, err_string);
        MPI_Abort(comm, mpi_err);
    }

    mpi_err = MPI_File_open(comm, (char*)output_path, MPI_MODE_CREATE | MPI_MODE_WRONLY, info_out, &output_file);
    MPI_Info_free(&info_in);
    MPI_Info_free(&info_out);
    if (mpi_err != MPI_SUCCESS) {
        char err_string[MPI_MAX_ERROR_STRING];
        int err_len = 0;
        MPI_Error_string(mpi_err, err_string, &err_len);
        fprintf(stderr, "[Rank %d] Failed to open output file '%s': %.*s\n", rank, output_path, err_len, err_string);
        MPI_Abort(comm, mpi_err);
    }

    if (rank == 0) {
        BinaryHeader header = {out_H, out_W};
        MPI_File_write_at(output_file, 0, &header, sizeof(BinaryHeader), MPI_BYTE, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(comm);
    uint32_t max_input_rows = chunk_rows * sH + kH;
    if (max_input_rows > params->H) max_input_rows = params->H;

    size_t max_input_elems = (size_t)max_input_rows * (size_t)W;
    size_t max_output_elems = (size_t)chunk_rows * (size_t)out_W;
    if (!max_output_elems) max_output_elems = (size_t)out_W;

    float* input_buf[2] = {alloc_aligned(max_input_elems), alloc_aligned(max_input_elems)};
    float* output_buf[2] = {alloc_aligned(max_output_elems), alloc_aligned(max_output_elems)};

    if (!input_buf[0] || !input_buf[1] || !output_buf[0] || !output_buf[1]) {
        fprintf(stderr, "[Rank %d] Failed to allocate double buffers\n", rank);
        if (input_buf[0]) free(input_buf[0]);
        if (input_buf[1]) free(input_buf[1]);
        if (output_buf[0]) free(output_buf[0]);
        if (output_buf[1]) free(output_buf[1]);
        MPI_Abort(comm, 1);
    }

    MPI_Request read_req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request write_req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    Chunk block[2] = {{0}};

    uint32_t scheduled = 0;
    uint32_t completed = 0;
    uint32_t next_start = row_start;
    int slot = 0;

    if (next_start < row_end) {
        build_chunk(&block[slot], next_start, chunk_rows, row_end, sH, kH, params->H, W, out_W);
        size_t need_input = (size_t)block[slot].num_input_rows * (size_t)W;
        if (need_input > max_input_elems) {
            fprintf(stderr, "[Rank %d] Input buffer too small (%zu > %zu)\n", rank, need_input, max_input_elems);
            MPI_Abort(comm, 1);
        }
        int count = (int)need_input;
        MPI_File_iread_at(input_file,
                          block[slot].input_offset,
                          input_buf[slot],
                          count,
                          MPI_FLOAT,
                          &read_req[slot]);
        scheduled++;
        next_start = block[slot].chunk_end;
    }

    while (completed < chunk_total) {
        MPI_Wait(&read_req[slot], MPI_STATUS_IGNORE);

        double t_chunk_start = MPI_Wtime();
        Chunk* info = &block[slot];

        size_t need_output = (size_t)info->chunk_out_H * (size_t)out_W;
        if (need_output > max_output_elems) {
            fprintf(stderr, "[Rank %d] Output buffer too small (%zu > %zu)\n", rank, need_output, max_output_elems);
            MPI_Abort(comm, 1);
        }

        ConvParams chunk_params = {
            .data = input_buf[slot],
            .kernel = params->kernel,
            .output = output_buf[slot],
            .H = info->num_input_rows,
            .W = W,
            .kH = kH,
            .kW = kW,
            .sH = sH,
            .sW = sW,
            .out_H = info->chunk_out_H,
            .out_W = out_W,
            .input_offset_row = info->input_row_start,
            .output_offset_row = info->chunk_start
        };

        double t_conv_start = MPI_Wtime();
        conv_openmp(&chunk_params);
        double t_conv = MPI_Wtime() - t_conv_start;

        int write_count = (int)need_output;
        MPI_File_iwrite_at(output_file,
                           info->output_offset,
                           output_buf[slot],
                           write_count,
                           MPI_FLOAT,
                           &write_req[slot]);

        completed++;

        if (scheduled < chunk_total && next_start < row_end) {
            int next_idx = slot ^ 1;
            if (write_req[next_idx] != MPI_REQUEST_NULL) {
                MPI_Wait(&write_req[next_idx], MPI_STATUS_IGNORE);
                write_req[next_idx] = MPI_REQUEST_NULL;
            }

            build_chunk(&block[next_idx], next_start, chunk_rows, row_end, sH, kH, params->H, W, out_W);
            size_t need_input_next = (size_t)block[next_idx].num_input_rows * (size_t)W;
            if (need_input_next > max_input_elems) {
                fprintf(stderr, "[Rank %d] Input buffer too small (%zu > %zu)\n", rank, need_input_next, max_input_elems);
                MPI_Abort(comm, 1);
            }
            int next_count = (int)need_input_next;
            MPI_File_iread_at(input_file,
                              block[next_idx].input_offset,
                              input_buf[next_idx],
                              next_count,
                              MPI_FLOAT,
                              &read_req[next_idx]);
            scheduled++;
            next_start = block[next_idx].chunk_end;
        }

        double t_chunk_total = MPI_Wtime() - t_chunk_start;
        printf("[MPI] rank=%d chunk=%u/%u out_rows=%u-%u in_rows=%u mem=%.1fMB time=%.4fs (io=%.4fs conv=%.4fs)\n",
               rank,
               completed,
               chunk_total,
               info->chunk_start,
               info->chunk_end,
               info->num_input_rows,
               ((double)info->num_input_rows * W + info->chunk_out_H * out_W) * sizeof(float) / 1e6,
               t_chunk_total,
               t_chunk_total - t_conv,
               t_conv);

        if (completed < chunk_total) {
            slot ^= 1;
        }
    }

    for (int i = 0; i < 2; ++i) {
        if (write_req[i] != MPI_REQUEST_NULL) {
            MPI_Wait(&write_req[i], MPI_STATUS_IGNORE);
        }
        if (read_req[i] != MPI_REQUEST_NULL) {
            MPI_Wait(&read_req[i], MPI_STATUS_IGNORE);
        }
        free(input_buf[i]);
        free(output_buf[i]);
    }

    MPI_File_close(&input_file);
    MPI_File_close(&output_file);
}
