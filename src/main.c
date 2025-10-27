#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>
#include "file.h"
#include "generate.h"
#include "conv.h"
#include "cli_parse.h"

static int ends_with(const char* s, const char* suf) {
    size_t n = strlen(s), m = strlen(suf);
    return n>=m && strcmp(s+n-m, suf)==0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world=1, rank=0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CLIArgs args = {-1, -1, -1, -1, 1, 1, NULL, NULL, NULL, 32.0, 0};
    
    if (rank == 0) {
        int parse_rc = parse_cli_args(argc, argv, &args);
        if (args.show_help) {
            print_cli_usage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
        if (parse_rc != 0) {
            print_cli_usage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    int cfg[6] = {args.H, args.W, args.kH, args.kW, args.sH, args.sW};
    MPI_Bcast(cfg, 6, MPI_INT, 0, MPI_COMM_WORLD);
    
    int H = cfg[0];
    int W = cfg[1];
    int kH = cfg[2];
    int kW = cfg[3];
    int sH = cfg[4];
    int sW = cfg[5];
    
    double mem_gb = args.memory_gb;
    MPI_Bcast(&mem_gb, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    char in_path_buf[256] = {0};
    char ker_path_buf[256] = {0};
    char out_path_buf[256] = {0};
    
    if (rank == 0) {
        if (args.input_file) strncpy(in_path_buf, args.input_file, 255);
        if (args.kernel_file) strncpy(ker_path_buf, args.kernel_file, 255);
        if (args.output_file) strncpy(out_path_buf, args.output_file, 255);
    }
    
    MPI_Bcast(in_path_buf, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(ker_path_buf, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(out_path_buf, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    const char* in_path = (in_path_buf[0] != '\0') ? in_path_buf : NULL;
    const char* ker_path = (ker_path_buf[0] != '\0') ? ker_path_buf : NULL;
    const char* out_path = out_path_buf;

    const char* tmp_dir = getenv("CONV_TEMP_DIR");
    if (!tmp_dir) tmp_dir = getenv("CONV_TMP_DIR");
    if (!tmp_dir) tmp_dir = "./tmp";
    if (rank == 0) {
        mkdir(tmp_dir, 0777);
    }
    
    char tmp_input_bin[256] = {0};
    int cleanup_input = 0;
    if (in_path && ends_with(in_path, ".txt")) {
        if (rank==0) {
            snprintf(tmp_input_bin, sizeof(tmp_input_bin), "%s/conv_input_%d.bin", tmp_dir, (int)getpid());
            convert_txt_to_bin((char*)in_path, tmp_input_bin, 8192);
        }
        MPI_Bcast(tmp_input_bin, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
        in_path = tmp_input_bin;
        cleanup_input = 1;
    }

    if (in_path) {
        if (rank==0) {
            BinaryFile bf = open_bin_matrix_input((char*)in_path);
            H = (int)bf.height; W = (int)bf.width;
            fclose(bf.file);
            
            cfg[0] = H;
            cfg[1] = W;
        }
        MPI_Bcast(cfg, 6, MPI_INT, 0, MPI_COMM_WORLD);
        H = cfg[0];
        W = cfg[1];
    }
    
    if (H<=0 || W<=0) { if (rank==0) fprintf(stderr, "Input size invalid or missing (-H -W or -f).\n"); MPI_Finalize(); return 2; }

    if (!in_path) {
        if (rank==0) {
            snprintf(tmp_input_bin, sizeof(tmp_input_bin), "%s/conv_input_%d.bin", tmp_dir, (int)getpid());
            generate_matrix_bin(tmp_input_bin, (uint32_t)H, (uint32_t)W, 1234);
        }
        MPI_Bcast(tmp_input_bin, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
        in_path = tmp_input_bin;
        cleanup_input = 1;
    }

    if (!ker_path && kH <= 0 && kW <= 0) {
        kH = 1;
        kW = 1;
        cfg[2] = kH;
        cfg[3] = kW;
        if (rank==0) {
            fprintf(stderr, "No kernel file or dimensions provided, assuming 1x1 identity kernel for matrix generation\n");
        }
    }

    char tmp_kernel_bin[256] = {0};
    int cleanup_kernel = 0;
    float* kernel_mem = NULL;
    if (ker_path && ends_with(ker_path, ".txt")) {
        if (rank==0) {
            snprintf(tmp_kernel_bin, sizeof(tmp_kernel_bin), "%s/conv_kernel_%d.bin", tmp_dir, (int)getpid());
            convert_txt_to_bin((char*)ker_path, tmp_kernel_bin, 8192);
        }
        MPI_Bcast(tmp_kernel_bin, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
        ker_path = tmp_kernel_bin;
        cleanup_kernel = 1;
    }
    
    if (ker_path && (kH <= 0 || kW <= 0)) {
        if (rank==0) {
            BinaryFile kb = open_bin_matrix_input((char*)ker_path);
            kH = (int)kb.height; kW = (int)kb.width;
            fclose(kb.file);
            
            cfg[2] = kH;
            cfg[3] = kW;
        }
        MPI_Bcast(cfg, 6, MPI_INT, 0, MPI_COMM_WORLD);
        kH = cfg[2];
        kW = cfg[3];
    }
    
    if (!ker_path) {
        if (rank==0) {
            kernel_mem = (float*)malloc((size_t)kH*(size_t)kW*sizeof(float));
            unsigned seed = 2025u;
            for (int i=0;i<kH*kW;++i) kernel_mem[i] = (float)(rand_r(&seed)%101)/100.0f;
        }
        if (!kernel_mem) kernel_mem = (float*)malloc((size_t)kH*(size_t)kW*sizeof(float));
        MPI_Bcast(kernel_mem, kH*kW, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    int use_mpi = (world > 1);

    const char* mem_env = getenv("CONV_MEM_GB");
    if (mem_env && atof(mem_env) > 0.0) mem_gb = atof(mem_env);
    double budget_bytes = mem_gb * 1024.0 * 1024.0 * 1024.0;

    const char* convert_env = getenv("CONVERT_BIN");
    int convert_to_txt = 1;
    if (convert_env && (strcmp(convert_env, "0") == 0 || strcmp(convert_env, "false") == 0 || strcmp(convert_env, "False") == 0)) {
        convert_to_txt = 0;
    }
    
    char tmp_output_bin[256] = {0};
    char final_output_path[256] = {0};
    int convert_output = 0;
    const char* internal_out = NULL;
    
    if (rank==0) {
        if (convert_to_txt) {
            snprintf(tmp_output_bin, sizeof(tmp_output_bin), "%s/conv_output_%d.bin", tmp_dir, (int)getpid());
            internal_out = tmp_output_bin;
            convert_output = 1;
        } else {
            if (!ends_with(out_path, ".bin")) {
                snprintf(final_output_path, sizeof(final_output_path), "%s.bin", out_path);
                internal_out = final_output_path;
            } else {
                internal_out = out_path;
            }
        }
    }
    
    if (convert_to_txt) {
        MPI_Bcast(tmp_output_bin, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
        internal_out = tmp_output_bin;
    } else {
        if (rank == 0 && final_output_path[0] != '\0') {
            strncpy(tmp_output_bin, final_output_path, 255);
        } else if (rank == 0) {
            strncpy(tmp_output_bin, out_path, 255);
        }
        MPI_Bcast(tmp_output_bin, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
        internal_out = tmp_output_bin;
    }

    double t0 = MPI_Wtime();

    int rc = 0;
    if (use_mpi) {
        float* kptr = NULL;
        if (kernel_mem) kptr = kernel_mem;
        else if (ker_path) {
            if (rank==0) {
                BinaryFile kb = open_bin_matrix_input((char*)ker_path);
                float* tmp = (float*)malloc((size_t)kH*(size_t)kW*sizeof(float));
                fread(tmp, sizeof(float), (size_t)kH*(size_t)kW, kb.file);
                fclose(kb.file);
                if (!kernel_mem) kernel_mem = tmp; else free(tmp);
            }
            if (!kernel_mem) kernel_mem = (float*)malloc((size_t)kH*(size_t)kW*sizeof(float));
            MPI_Bcast(kernel_mem, kH*kW, MPI_FLOAT, 0, MPI_COMM_WORLD);
            kptr = kernel_mem;
        }

        ConvParams* mpi_params = (ConvParams*)malloc(sizeof(ConvParams));
        
        mpi_params->H = (uint32_t)H;
        mpi_params->W = (uint32_t)W;
        mpi_params->kH = (uint32_t)kH;
        mpi_params->kW = (uint32_t)kW;
        mpi_params->sH = (uint32_t)sH;
        mpi_params->sW = (uint32_t)sW;
        mpi_params->kernel = kptr;
        mpi_params->data = NULL;
        mpi_params->output = NULL;  
        mpi_params->input_offset_row = 0;
        mpi_params->output_offset_row = 0;
        calc_output_dims(mpi_params);
        
        conv_mpi(mpi_params, MPI_COMM_WORLD, in_path, internal_out, budget_bytes);
        
        free(mpi_params);
        rc = 0;
    } else {
        float* kernel_full = NULL;

        if (rank==0) {
            if (kernel_mem) kernel_full = kernel_mem;
            else if (ker_path) {
                BinaryFile kb = open_bin_matrix_input((char*)ker_path);
                kernel_full = (float*)malloc((size_t)kH*(size_t)kW*sizeof(float));
                fread(kernel_full, sizeof(float), (size_t)kH*(size_t)kW, kb.file);
                fclose(kb.file);
            }
            
            int out_H = calc_output_height(H, kH, sH);
            int out_W = calc_output_width(W, kW, sW);
            
            uint32_t chunk_out_rows = calc_chunk_size((uint32_t)W, (uint32_t)out_W, 
                                                       (uint32_t)kH, (uint32_t)kW, 
                                                       (uint32_t)sH, (size_t)budget_bytes);
            
            uint32_t num_chunks = (out_H + chunk_out_rows - 1) / chunk_out_rows;
            
            size_t chunk_mem_size = (size_t)chunk_out_rows * (sH + kH) * W * sizeof(float) + 
                                   (size_t)chunk_out_rows * out_W * sizeof(float);
            uint32_t max_chunks_in_mem = (uint32_t)(budget_bytes / chunk_mem_size);
            if (max_chunks_in_mem < 1) max_chunks_in_mem = 1;
            
            fprintf(stdout,"[CHUNK] mode=%s threads=%s mem=%.3fGB chunk_rows=%u total_chunks=%u max_in_mem=%u out_size=%dx%d\n",
                   "omp",
                   getenv("OMP_NUM_THREADS")?getenv("OMP_NUM_THREADS"):"1",
                   budget_bytes/1e9, chunk_out_rows, num_chunks, max_chunks_in_mem, out_H, out_W);
            
            double t_read_done = MPI_Wtime();
            double t_comp_total = 0.0;
            uint32_t chunk_counter = 0;
            
            FILE* output_file = create_bin_matrix((char*)internal_out, (uint32_t)out_H, (uint32_t)out_W);
            
            typedef struct {
                float* input;
                float* output;
                uint32_t out_row_start;
                uint32_t out_row_end;
                uint32_t input_row_start;
                uint32_t num_input_rows;
                int loaded;
                int processed;
            } ChunkBuffer;
            
            ChunkBuffer* buffers = (ChunkBuffer*)calloc(max_chunks_in_mem, sizeof(ChunkBuffer));
            
            uint32_t next_chunk_to_load = 0;
            uint32_t next_chunk_to_process = 0;
            uint32_t chunks_in_memory = 0;
            
            while (next_chunk_to_process < num_chunks) {
                while (chunks_in_memory < max_chunks_in_mem && next_chunk_to_load < num_chunks) {
                    uint32_t out_row_start = next_chunk_to_load * chunk_out_rows;
                    uint32_t out_row_end = out_row_start + chunk_out_rows;
                    if (out_row_end > (uint32_t)out_H) out_row_end = (uint32_t)out_H;
                    
                    uint32_t input_row_start, num_input_rows;
                    calc_input_rows_for_output_range_clamped(out_row_start, out_row_end, (uint32_t)sH, (uint32_t)kH, (uint32_t)H,
                                                              &input_row_start, &num_input_rows);
                    
                    uint32_t chunk_out_H = out_row_end - out_row_start;
                    uint32_t buf_idx = next_chunk_to_load % max_chunks_in_mem;
                    
                    buffers[buf_idx].input = (float*)malloc((size_t)num_input_rows * W * sizeof(float));
                    buffers[buf_idx].output = (float*)malloc((size_t)chunk_out_H * out_W * sizeof(float));
                    BinaryFile input_file = open_bin_matrix_input((char*)in_path);
                    fseek(input_file.file, sizeof(BinaryHeader) + (size_t)input_row_start * W * sizeof(float), SEEK_SET);
                    fread(buffers[buf_idx].input, sizeof(float), (size_t)num_input_rows * W, input_file.file);
                    fclose(input_file.file);
                    
                    buffers[buf_idx].out_row_start = out_row_start;
                    buffers[buf_idx].out_row_end = out_row_end;
                    buffers[buf_idx].input_row_start = input_row_start;
                    buffers[buf_idx].num_input_rows = num_input_rows;
                    buffers[buf_idx].loaded = 1;
                    buffers[buf_idx].processed = 0;
                    
                    next_chunk_to_load++;
                    chunks_in_memory++;
                }
                
                uint32_t buf_idx = next_chunk_to_process % max_chunks_in_mem;
                if (!buffers[buf_idx].loaded) break;
                
                chunk_counter++;
                double t_chunk_start = MPI_Wtime();
                
                uint32_t chunk_out_H = buffers[buf_idx].out_row_end - buffers[buf_idx].out_row_start;
                
                ConvParams chunk_params = {
                    .data = buffers[buf_idx].input,
                    .kernel = kernel_full,
                    .output = buffers[buf_idx].output,
                    .H = buffers[buf_idx].num_input_rows,
                    .W = (uint32_t)W,
                    .kH = (uint32_t)kH,
                    .kW = (uint32_t)kW,
                    .sH = (uint32_t)sH,
                    .sW = (uint32_t)sW,
                    .out_H = chunk_out_H,
                    .out_W = (uint32_t)out_W,
                    .input_offset_row = buffers[buf_idx].input_row_start,
                    .output_offset_row = buffers[buf_idx].out_row_start
                };
                
                double t_conv_start = MPI_Wtime();
                conv_openmp(&chunk_params);
                double t_conv = MPI_Wtime() - t_conv_start;
                t_comp_total += t_conv;
                
                fseek(output_file, sizeof(BinaryHeader) + (size_t)buffers[buf_idx].out_row_start * out_W * sizeof(float), SEEK_SET);
                fwrite(buffers[buf_idx].output, sizeof(float), (size_t)chunk_out_H * out_W, output_file);
                
                double t_chunk_total = MPI_Wtime() - t_chunk_start;
                fprintf(stdout,"[CHUNK] %u/%u out_rows=%u-%u in_rows=%u mem=%.1fMB chunks_loaded=%u time=%.4fs (io=%.4fs conv=%.4fs)\n",
                       chunk_counter, num_chunks, buffers[buf_idx].out_row_start, buffers[buf_idx].out_row_end, 
                       buffers[buf_idx].num_input_rows,
                       ((double)buffers[buf_idx].num_input_rows * W + chunk_out_H * out_W) * sizeof(float) / 1e6,
                       chunks_in_memory, t_chunk_total, t_chunk_total - t_conv, t_conv);
                
                free(buffers[buf_idx].input);
                free(buffers[buf_idx].output);
                buffers[buf_idx].loaded = 0;
                buffers[buf_idx].processed = 1;
                
                chunks_in_memory--;
                next_chunk_to_process++;
            }
            
            free(buffers);
            fclose(output_file);

            double t_all_done = MPI_Wtime();
            double t_read = t_read_done - t0;
            double t_comp = t_comp_total;
            double t_write = t_all_done - t_read_done - t_comp_total;
            fprintf(stdout,"mode=%s ranks=%d threads=%s H=%d W=%d k=%dx%d s=%dx%d read=%.3fs comp=%.3fs write=%.3fs\n",
                   "omp", world, getenv("OMP_NUM_THREADS")?getenv("OMP_NUM_THREADS"):"1",
                   H,W,kH,kW,sH,sW,t_read,t_comp,t_write);

            if (!kernel_mem && kernel_full) free(kernel_full);
        }
    }

    if (use_mpi) {
        double t_done = MPI_Wtime();
        if (rank==0) {
            printf("mode=mpi ranks=%d threads=%s H=%d W=%d k=%dx%d s=%dx%d total=%.3fs\n",
                   world, getenv("OMP_NUM_THREADS")?getenv("OMP_NUM_THREADS"):"?",
                   H,W,kH,kW,sH,sW, t_done - t0);
        }
    }

    if (rank==0 && convert_to_txt && convert_output) {
        convert_bin_to_txt((char*)internal_out, (char*)out_path, 8192);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank==0) {
        if (cleanup_input && tmp_input_bin[0]) {
            remove(tmp_input_bin);
        }
        if (cleanup_kernel && tmp_kernel_bin[0]) {
            remove(tmp_kernel_bin);
        }
        if (convert_output && tmp_output_bin[0]) {
            remove(tmp_output_bin);
        }
    }

    if (kernel_mem) free(kernel_mem);
    MPI_Finalize();
    return rc;
}
