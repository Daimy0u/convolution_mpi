#include "cli_parse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

void print_cli_usage(const char* program_name) {
    fprintf(stderr, "Usage: %s [OPTIONS]\n", program_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -H, --height=N        Input matrix height (required if no -f)\n");
    fprintf(stderr, "  -W, --width=N         Input matrix width (required if no -f)\n");
    fprintf(stderr, "  -kH,                  Kernel height (required if no -g)\n");
    fprintf(stderr, "  -kW,                  Kernel width (required if no -g)\n");
    fprintf(stderr, "  -sH,                  Vertical stride (default: 1)\n");
    fprintf(stderr, "  -sW,                  Horizontal stride (default: 1)\n");
    fprintf(stderr, "  -f, --input=FILE      Input matrix file (.txt or .bin)\n");
    fprintf(stderr, "  -g, --kernel=FILE     Kernel file (.txt or .bin)\n");
    fprintf(stderr, "  -o, --output=FILE     Output file (required)\n");
    fprintf(stderr, "  -M, --memory=GB       Memory budget in GB (default: 32.0)\n");
    fprintf(stderr, "  -h, --help            Display this help message\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s -H 1000 -W 1000 -kH 5 -kW 5 -o output.bin\n", program_name);
    fprintf(stderr, "  %s -f input.txt -g kernel.txt -sH 2 -sW 2 -o output.bin\n", program_name);
    fprintf(stderr, "  %s --input=input.bin --kernel=kernel.bin -kH 10 -kW 10 -M 16 -o out.bin\n", program_name);
}

static int parse_int_arg(const char* s) {
    if (!s || !*s) return -1;
    char* end = NULL;
    long value = strtol(s, &end, 10);
    if (end && *end) return -1;
    if (value < 0 || value > 0x7fffffffL) return -1;
    return (int)value;
}

static double parse_double_arg(const char* s) {
    if (!s || !*s) return -1.0;
    char* end = NULL;
    double value = strtod(s, &end);
    if (end && *end) return -1.0;
    return (value > 0.0) ? value : -1.0;
}

static char** expand_short_flags(int argc, char** argv, int* fixed_argc) {
    char** list = (char**)malloc((size_t)argc * sizeof(char*));
    if (!list) return NULL;

    *fixed_argc = argc;
    for (int i = 0; i < argc; ++i) {
        if (argv[i][0] == '-' && argv[i][1] != '-' &&
            (strncmp(argv[i], "-kH", 3) == 0 ||
             strncmp(argv[i], "-kW", 3) == 0 ||
             strncmp(argv[i], "-sH", 3) == 0 ||
             strncmp(argv[i], "-sW", 3) == 0)) {

            size_t len = strlen(argv[i]);
            list[i] = (char*)malloc(len + 2);
            if (!list[i]) {
                for (int j = 0; j < i; ++j) {
                    if (list[j] != argv[j]) free(list[j]);
                }
                free(list);
                return NULL;
            }
            list[i][0] = '-';
            strcpy(list[i] + 1, argv[i]);
        } else {
            list[i] = argv[i];
        }
    }

    return list;
}

static void free_expanded_args(int argc, char** list, char** original) {
    for (int i = 0; i < argc; ++i) {
        if (list[i] != original[i]) {
            free(list[i]);
        }
    }
    free(list);
}

int parse_cli_args(int argc, char** argv, CLIArgs* args) {
    args->H = args->W = -1;
    args->kH = args->kW = -1;
    args->sH = args->sW = 1;
    args->input_file = NULL;
    args->kernel_file = NULL;
    args->output_file = NULL;
    args->memory_gb = 8.0;
    args->show_help = 0;

    int fixed_argc = 0;
    char** fixed_argv = expand_short_flags(argc, argv, &fixed_argc);
    if (!fixed_argv) {
        fprintf(stderr, "Error: Memory allocation failed during argument preprocessing\n");
        return 1;
    }

    static struct option long_options[] = {
        {"height",  required_argument, 0, 'H'},
        {"width",   required_argument, 0, 'W'},
        {"kH",      required_argument, 0, 'a'},
        {"kW",      required_argument, 0, 'b'},
        {"sH",      required_argument, 0, 'c'},
        {"sW",      required_argument, 0, 'd'},
        {"input",   required_argument, 0, 'f'},
        {"kernel",  required_argument, 0, 'g'},
        {"output",  required_argument, 0, 'o'},
        {"memory",  required_argument, 0, 'M'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int flag = 0;

    while ((flag = getopt_long(fixed_argc, fixed_argv, "H:W:f:g:o:M:h", long_options, &option_index)) != -1) {
        switch (flag) {
            case 'H':
                args->H = parse_int_arg(optarg);
                if (args->H < 0) {
                    fprintf(stderr, "Error: Invalid height value: %s\n", optarg);
                    free_expanded_args(fixed_argc, fixed_argv, argv);
                    return 1;
                }
                break;
            case 'W':
                args->W = parse_int_arg(optarg);
                if (args->W < 0) {
                    fprintf(stderr, "Error: Invalid width value: %s\n", optarg);
                    free_expanded_args(fixed_argc, fixed_argv, argv);
                    return 1;
                }
                break;
            case 'a':
                args->kH = parse_int_arg(optarg);
                if (args->kH < 0) {
                    fprintf(stderr, "Error: Invalid kernel height value: %s\n", optarg);
                    free_expanded_args(fixed_argc, fixed_argv, argv);
                    return 1;
                }
                break;
            case 'b':
                args->kW = parse_int_arg(optarg);
                if (args->kW < 0) {
                    fprintf(stderr, "Error: Invalid kernel width value: %s\n", optarg);
                    free_expanded_args(fixed_argc, fixed_argv, argv);
                    return 1;
                }
                break;
            case 'c':
                args->sH = parse_int_arg(optarg);
                if (args->sH < 0) {
                    fprintf(stderr, "Error: Invalid stride height value: %s\n", optarg);
                    free_expanded_args(fixed_argc, fixed_argv, argv);
                    return 1;
                }
                break;
            case 'd':
                args->sW = parse_int_arg(optarg);
                if (args->sW < 0) {
                    fprintf(stderr, "Error: Invalid stride width value: %s\n", optarg);
                    free_expanded_args(fixed_argc, fixed_argv, argv);
                    return 1;
                }
                break;
            case 'f':
                args->input_file = optarg;
                break;
            case 'g':
                args->kernel_file = optarg;
                break;
            case 'o':
                args->output_file = optarg;
                break;
            case 'M':
                args->memory_gb = parse_double_arg(optarg);
                if (args->memory_gb < 0.0) {
                    fprintf(stderr, "Error: Invalid memory budget value: %s\n", optarg);
                    free_expanded_args(fixed_argc, fixed_argv, argv);
                    return 1;
                }
                break;
            case 'h':
                args->show_help = 1;
                free_expanded_args(fixed_argc, fixed_argv, argv);
                return 0;
            case '?':
            default:
                free_expanded_args(fixed_argc, fixed_argv, argv);
                return 1;
        }
    }

    if (optind < fixed_argc) {
        fprintf(stderr, "Error: Unexpected argument: %s\n", fixed_argv[optind]);
        free_expanded_args(fixed_argc, fixed_argv, argv);
        return 1;
    }

    if ((args->kH <= 0 || args->kW <= 0) && !args->kernel_file) {
        fprintf(stderr, "Error: Kernel dimensions (-kH and -kW) are required unless kernel file (-g) is provided\n");
        free_expanded_args(fixed_argc, fixed_argv, argv);
        return 1;
    }

    if (!args->output_file) {
        fprintf(stderr, "Error: Output file (-o/--output) is required\n");
        free_expanded_args(fixed_argc, fixed_argv, argv);
        return 1;
    }

    if (!args->input_file && (args->H <= 0 || args->W <= 0)) {
        fprintf(stderr, "Error: Either input file (-f) or dimensions (-H and -W) must be specified\n");
        free_expanded_args(fixed_argc, fixed_argv, argv);
        return 1;
    }

    free_expanded_args(fixed_argc, fixed_argv, argv);
    return 0;
}
