#ifndef CLI_PARSE_H
#define CLI_PARSE_H

#include <stdint.h>

typedef struct {
    int H;
    int W;
    int kH;
    int kW;
    int sH;
    int sW;
    const char* input_file;
    const char* kernel_file;
    const char* output_file;
    double memory_gb;
    int show_help;
} CLIArgs;

int parse_cli_args(int argc, char** argv, CLIArgs* args);
void print_cli_usage(const char* program_name);

#endif // CLI_PARSE_H

