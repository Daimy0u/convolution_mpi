#include "generate.h"
#include "file.h"

void generate_matrix_bin(char* bin_fp, uint32_t h, uint32_t w, uint32_t seed) {
    int chunk = 10000;
    uint64_t written = 0;

    seed ? (uint32_t)seed : (uint32_t)time(NULL);
    srand(seed);

    FILE* bin = create_bin_matrix(bin_fp, h, w);
    fseek(bin, sizeof(BinaryHeader), SEEK_SET);

    float* buf = (float*)malloc((size_t)chunk * sizeof(float));
    
    #pragma omp parallel for firstprivate(buf) reduction(+ : written)
    for (int n = 0; n < (int)(h * w); n++) {
        int made = 0;
        for (int i = 0; (i < chunk) && (i < (int)(h * w) - (int)written); i++) {
            buf[i] = (rand() % 101) / 100.0f;
            made++;
        }
        
        fwrite(buf, sizeof(float), made, bin);
        written += (uint64_t)made;
    }
    fclose(bin);
    free(buf);
}
