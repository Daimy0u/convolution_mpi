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

    int use_mpi = 1;

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
