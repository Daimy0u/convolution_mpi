#include "file.h"
#include <math.h>
#include <sys/types.h>
#include <fenv.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <ctype.h>
#include <stdatomic.h>

#if defined(__unix__) || defined(__APPLE__)
ssize_t pwrite(int fd, const void* buf, size_t nbyte, off_t offset);
#endif

static ssize_t read_line(FILE* stream, char** buffer, size_t* capacity) {
    if (!stream || !buffer || !capacity) {
        errno = EINVAL;
        return -1;
    }

    if (*buffer == NULL || *capacity == 0) {
        size_t new_cap = 256;
        char* tmp = (char*)malloc(new_cap);
        if (!tmp) {
            errno = ENOMEM;
            return -1;
        }
        *buffer = tmp;
        *capacity = new_cap;
    }

    size_t pos = 0;
    int ch = 0;

    while ((ch = fgetc(stream)) != EOF) {
        if (pos + 1 >= *capacity) {
            size_t new_cap = (*capacity < SIZE_MAX / 2) ? (*capacity * 2) : SIZE_MAX;
            if (new_cap == *capacity) {
                errno = ENOMEM;
                return -1;
            }
            char* tmp = (char*)realloc(*buffer, new_cap);
            if (!tmp) {
                errno = ENOMEM;
                return -1;
            }
            *buffer = tmp;
            *capacity = new_cap;
        }

        (*buffer)[pos++] = (char)ch;
        if (ch == '\n') {
            break;
        }
    }

    if (pos == 0 && ch == EOF) {
        return -1;
    }

    (*buffer)[pos] = '\0';
    return (ssize_t)pos;
}

static ssize_t write_at_pos(int fd, const void* buffer, size_t bytes, off_t offset) {
    ssize_t direct = -1;
#if defined(__unix__) || defined(__APPLE__)
    errno = 0;
    direct = pwrite(fd, buffer, bytes, offset);
    if (!(direct == -1 && (errno == ENOSYS || errno == EINVAL))) {
        return direct;
    }
#endif

    ssize_t written = -1;
    int saved_errno = 0;
    #pragma omp critical(write_at_fallback)
    {
        off_t current = lseek(fd, 0, SEEK_CUR);
        if (lseek(fd, offset, SEEK_SET) == (off_t)-1) {
            saved_errno = errno;
            written = -1;
        } else {
            written = write(fd, buffer, bytes);
            saved_errno = errno;
        }
        if (current != (off_t)-1) {
            lseek(fd, current, SEEK_SET);
        }
    }
    errno = saved_errno;
    return written;
}

FILE* create_bin_matrix(char* filepath, uint32_t h, uint32_t w) {
    BinaryHeader header = {h, w};
    const uint64_t total_elements = (uint64_t)h * (uint64_t)w;
    const size_t header_size = sizeof(header);

    FILE* o_file = fopen(filepath, "wb+");

    if (!o_file) return NULL;

    if (fwrite(&header, header_size, 1, o_file) != 1) goto fail;
    if (fflush(o_file) != 0) goto fail;

    if (total_elements > 0) {
        int fd = fileno(o_file);
        if (fd == -1) goto fail;

        const size_t block_elems = 32768; 
        _Atomic int error_flag = 0;

        #pragma omp parallel
        {
            float* zero_block = (float*)calloc(block_elems, sizeof(float));

            #pragma omp for schedule(static)
            for (uint64_t start = 0; start < total_elements; start += block_elems) {
                if (atomic_load_explicit(&error_flag, memory_order_relaxed)) continue;
                

                size_t remaining = (size_t)(total_elements - start);
                size_t count = remaining < block_elems ? remaining : block_elems;

                const size_t bytes = count * sizeof(float);
                off_t offset = (off_t)header_size + (off_t)(start * sizeof(float));

                const float* buffer = zero_block;
                float stack_block[1024] = {0};

                if (!buffer) {
                    size_t local_count = count < 1024 ? count : 1024;
                    buffer = stack_block;
                    // write in slices to avoid overrunning the stack buffer
                    size_t remaining_bytes = bytes;
                    size_t processed = 0;
                    while (remaining_bytes > 0) {
                        size_t slice_count = remaining_bytes / sizeof(float);
                        if (slice_count > local_count) slice_count = local_count;
                        size_t slice_bytes = slice_count * sizeof(float);
                        off_t slice_offset = offset + (off_t)(processed * sizeof(float));
                        ssize_t written = write_at_pos(fd, buffer, (size_t)slice_bytes, slice_offset);
                        if (written != (ssize_t)slice_bytes) {
                            #pragma omp critical
                            {
                                if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                                    fprintf(stderr, "Failed to initialise payload for %s (%s)\n", filepath, strerror(errno));
                                    atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                                }
                            }
                            break;
                        }
                        remaining_bytes -= slice_bytes;
                        processed += slice_count;
                    }
                    continue;
                }

                ssize_t written = write_at_pos(fd, buffer, bytes, offset);
                if (written != (ssize_t)bytes) {
                    #pragma omp critical
                    {
                        if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                            fprintf(stderr, "Failed to initialise payload for %s (%s)\n", filepath, strerror(errno));
                            atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                        }
                    }
                }
            }

            free(zero_block);
        }

        if (atomic_load_explicit(&error_flag, memory_order_relaxed)) {
            goto fail;
        }
    }

    if (fseeko(o_file, (off_t)header_size, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to rewind payload in %s (%s)\n", filepath, strerror(errno));
        goto fail;
    }

    return o_file;

fail:
    fclose(o_file);
    remove(filepath);
    return NULL;
}

BinaryFile open_bin_matrix_input(char* filepath) {
    BinaryFile out = {0, 0, NULL};
    if (!filepath) {
        return out;
    }

    FILE* file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file for reading: %s (%s)\n", filepath, strerror(errno));
        return out;
    }

    BinaryHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        fprintf(stderr, "Failed to read header from %s (%s)\n", filepath, strerror(errno));
        fclose(file);
        return out;
    }

    const uint64_t elements = (uint64_t)header.height * (uint64_t)header.width;
    if (elements == 0) {
        fclose(file);
        return out;
    }

    off_t f_offset = (off_t)sizeof(header) + ((off_t)elements - 1) * (off_t)sizeof(float);
    if (fseeko(file, f_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to end of payload in %s (%s)\n", filepath, strerror(errno));
        fclose(file);
        return out;
    }

    float last;
    if (fread(&last, sizeof(last), 1, file) != 1) {
        fprintf(stderr, "Failed to read payload from %s (%s)\n", filepath, strerror(errno));
        fclose(file);
        return out;
    }

    if (fseeko(file, (off_t)sizeof(header), SEEK_SET) != 0) {
        fprintf(stderr, "Failed to rewind payload in %s (%s)\n", filepath, strerror(errno));
        fclose(file);
        return out;
    }

    out.height = header.height;
    out.width = header.width;
    out.file = file;
    return out;
}

void apply_padding_bin(char* bin_fp, char* dst_fp, MatrixPadding* padding, size_t chunk_size) {
    if (!bin_fp) return;
    if (!chunk_size) chunk_size = 5000;
    printf("Applying padding to %s \nProgress: ",bin_fp);

    BinaryFile b_in = open_bin_matrix_input(bin_fp);
    if (b_in.height == 0 || !b_in.file) {printf("ERROR"); return;}
    printf("FIO_S:OK, ");

    FILE* bin = b_in.file;
    uint32_t h = b_in.height, w = b_in.width;
    uint32_t Hp = h + padding->pad_h_a + padding->pad_h_b, Wp = w + padding->pad_w_a + padding->pad_w_b;

    FILE* bin_p = create_bin_matrix(dst_fp, Hp, Wp);
    if (!bin_p) {
        fclose(bin);
        return;
    }
    if (fseeko(bin_p,(off_t)sizeof(BinaryHeader),SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek padded file %s (%s)\n", dst_fp, strerror(errno));
        fclose(bin);
        fclose(bin_p);
        remove(dst_fp);
        return;
    }
    printf("FIO_D:OK, ");

    //one dimension should be fine up to 10**9 on local, 10**10 on HPC (40 gb though)
    float* pad_row = (Wp ? (float*)calloc((size_t)Wp, sizeof(float)) : NULL);
    float* pad_b = (padding->pad_w_b ? (float*)calloc((size_t)padding->pad_w_b, sizeof(float)) : NULL);
    float* pad_a = (padding->pad_w_a ? (float*)calloc((size_t)padding->pad_w_a, sizeof(float)) : NULL);

    float* row_buffer = (float *)malloc((size_t)w * sizeof(float));

    if ((Wp && !pad_row) || (padding->pad_w_b && !pad_b) || (padding->pad_w_a && !pad_a) || !row_buffer) {
        fprintf(stderr, "Failed to allocate padding buffers (%s)\n", strerror(errno));
        free(row_buffer);
        free(pad_a);
        free(pad_b);
        free(pad_row);
        fclose(bin);
        fclose(bin_p);
        remove(dst_fp);
        return;
    }

    printf("BUF:OK, ");
    uint32_t row_written = 0;
    for (int p = 0; p < padding->pad_h_b; p++) {
        fwrite(pad_row,sizeof(float),Wp,bin_p);
        row_written++;
    }
    while (fread(row_buffer,sizeof(float),w, bin) > 0) {
        if (padding->pad_w_b && fwrite(pad_b,sizeof(float), padding->pad_w_b, bin_p) != padding->pad_w_b) break;
        if (fwrite(row_buffer,sizeof(float),w, bin_p) != w) break;
        if (padding->pad_w_a && fwrite(pad_a,sizeof(float), padding->pad_w_a, bin_p) != padding->pad_w_a) break;
        row_written++;
    }
    for (int p = 0; p < padding->pad_h_a; p++) {
        fwrite(pad_row,sizeof(float),Wp,bin_p);
        row_written++;
    }
    if (row_written != Hp) {
        //delete file or something
        printf("T:INVALID, ROW COUNT MISMATCH!\n");
    } else {
        printf("T:OK!\n");
    }

    if (bin) { fclose(bin); }
    if (bin_p) { fclose(bin_p); }

    free(row_buffer);
    free(pad_a); free(pad_b); free(pad_row);
}


void convert_txt_to_bin(char* txt_fp, char* bin_fp, size_t chunk_size) {
    int h, w;
    if (!txt_fp || !bin_fp) return;

    FILE* txt = fopen(txt_fp, "r");
    if (!txt) {
        fprintf(stderr, "Failed to open text file %s (%s)\n", txt_fp, strerror(errno));
        return;
    }

    if (fscanf(txt, "%d %d", &h, &w) != 2) {
        fprintf(stderr, "Input Matrix has an invalid dimension header in %s\n", txt_fp);
        fclose(txt);
        return;
    }
    if (h <= 0 || w <= 0) {
        fprintf(stderr, "Invalid matrix dimensions %d x %d in %s\n", h, w, txt_fp);
        fclose(txt);
        return;
    }

    // Consume trailing newline from the dimension header to prepare for getline
    int c;
    while ((c = fgetc(txt)) != '\n') {
        if (c == EOF) {
            break;
        }
    }

    FILE* bin = create_bin_matrix(bin_fp, (uint32_t)h, (uint32_t)w);
    if (!bin) {
        fclose(txt);
        return;
    }

    if (fflush(bin) != 0) {
        fprintf(stderr, "Failed to flush binary output %s (%s)\n", bin_fp, strerror(errno));
        fclose(bin);
        fclose(txt);
        remove(bin_fp);
        return;
    }

    int fd = fileno(bin);
    if (fd == -1) {
        fprintf(stderr, "Failed to obtain file descriptor for %s (%s)\n", bin_fp, strerror(errno));
        fclose(bin);
        fclose(txt);
        remove(bin_fp);
        return;
    }

    size_t row_capacity = chunk_size ? chunk_size : (size_t)w;
    if (row_capacity < (size_t)w) {
        row_capacity = (size_t)w;
    }

    _Atomic int error_flag = 0;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            char* line = NULL;
            size_t linecap = 0;

            for (uint32_t row = 0; row < (uint32_t)h; ++row) {
                if (atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                    break;
                }

                errno = 0;
                ssize_t read_len = read_line(txt, &line, &linecap);
                if (read_len == -1) {
                    #pragma omp critical
                    {
                        if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                            fprintf(stderr, "Failed to read row %u from %s (%s)\n", row, txt_fp, strerror(errno));
                            atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                        }
                    }
                    break;
                }

                char* row_copy = (char*)malloc((size_t)read_len + 1);
                if (!row_copy) {
                    #pragma omp critical
                    {
                        if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                            fprintf(stderr, "Failed to allocate row buffer for %s (%s)\n", txt_fp, strerror(errno));
                            atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                        }
                    }
                    break;
                }
                memcpy(row_copy, line, (size_t)read_len + 1);

                #pragma omp task firstprivate(row_copy, row) shared(error_flag, fd, row_capacity, w, bin_fp, txt_fp)
                {
                    int skip_task = atomic_load_explicit(&error_flag, memory_order_relaxed);
                    float* values = NULL;

                    if (!skip_task) {
                        values = (float*)malloc(row_capacity * sizeof(float));
                        if (!values) {
                            #pragma omp critical
                            {
                                if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                                    fprintf(stderr, "Failed to allocate conversion buffer for %s (%s)\n", txt_fp, strerror(errno));
                                    atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                                }
                            }
                            skip_task = 1;
                        }
                    }

                    size_t count = 0;
                    if (!skip_task && values) {
                        char* cursor = row_copy;
                        while (*cursor != '\0' && count < (size_t)w) {
                            while (*cursor && isspace((unsigned char)*cursor)) {
                                cursor++;
                            }
                            if (*cursor == '\0') {
                                break;
                            }

                            errno = 0;
                            char* endptr = cursor;
                            float value = strtof(cursor, &endptr);
                            if (errno != 0 || endptr == cursor) {
                                #pragma omp critical
                                {
                                    if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                                        fprintf(stderr, "Failed to parse value in row %u of %s\n", row, txt_fp);
                                        atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                                    }
                                }
                                skip_task = 1;
                                break;
                            }

                            if (count >= row_capacity) {
                                #pragma omp critical
                                {
                                    if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                                        fprintf(stderr, "Row buffer overflow while processing %s\n", txt_fp);
                                        atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                                    }
                                }
                                skip_task = 1;
                                break;
                            }

                            values[count++] = value;
                            cursor = endptr;
                        }

                        if (!skip_task && count != (size_t)w) {
                            #pragma omp critical
                            {
                                if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                                    fprintf(stderr, "Row %u in %s expected %u values, found %zu\n", row, txt_fp, w, count);
                                    atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                                }
                            }
                            skip_task = 1;
                        }
                    }

                    if (!skip_task && values) {
                        off_t offset = (off_t)sizeof(BinaryHeader) +
                                       (off_t)row * (off_t)w * (off_t)sizeof(float);
                        size_t bytes = (size_t)w * sizeof(float);
                        ssize_t written = write_at_pos(fd, values, bytes, offset);
                        if (written != (ssize_t)bytes) {
                            #pragma omp critical
                            {
                                if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                                    fprintf(stderr, "Failed to write row %u to %s (%s)\n", row, bin_fp, strerror(errno));
                                    atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                                }
                            }
                        }
                    }

                    free(values);
                    free(row_copy);
                }
            }

            free(line);
            #pragma omp taskwait
        }
    }

    if (atomic_load_explicit(&error_flag, memory_order_relaxed)) {
        fclose(bin);
        fclose(txt);
        remove(bin_fp);
        return;
    }

    fclose(bin);
    fclose(txt);
}

void convert_bin_to_txt(char* bin_fp, char* txt_fp, size_t chunk_size) {
    if (!txt_fp || !bin_fp) return;
    BinaryFile b_in = open_bin_matrix_input(bin_fp);
    if (b_in.height == 0 || !b_in.file) {
        fprintf(stderr, "Failed to open binary matrix %s\n", bin_fp);
        return;
    }

    FILE* bin = b_in.file;
    uint32_t h = b_in.height, w = b_in.width;

    FILE* txt = fopen(txt_fp, "w+");
    if (!txt) {
        fprintf(stderr, "Failed to open text file %s (%s)\n", txt_fp, strerror(errno));
        fclose(bin);
        return;
    }
    fprintf(txt, "%d %d\n", h, w);

    uint64_t total_elements = (uint64_t)h * (uint64_t)w;
    if (total_elements == 0) {
        fclose(bin);
        fclose(txt);
        return;
    }

    if (!chunk_size) {
        chunk_size = 5000;
    }
    uint64_t chunk_elems = chunk_size;

    uint64_t chunk_count_u64 = (total_elements + chunk_elems - 1) / chunk_elems;
    if (chunk_count_u64 > SIZE_MAX) {
        fprintf(stderr, "Matrix too large to convert with chunked buffers: %s\n", bin_fp);
        fclose(bin);
        fclose(txt);
        remove(txt_fp);
        return;
    }
    size_t chunk_count = (size_t)chunk_count_u64;

    typedef struct {
        char* data;
        size_t length;
    } ChunkText;

    ChunkText* results = (ChunkText*)calloc(chunk_count, sizeof(ChunkText));
    if (!results) {
        fprintf(stderr, "Failed to allocate chunk bookkeeping for %s (%s)\n", txt_fp, strerror(errno));
        fclose(bin);
        fclose(txt);
        remove(txt_fp);
        return;
    }

    _Atomic int error_flag = 0;

    const size_t max_chars_per_value = 32;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            uint64_t processed = 0;
            size_t chunk_index = 0;

            while (processed < total_elements &&
                   !atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                uint64_t remaining = total_elements - processed;
                size_t current_chunk = (size_t)(remaining < chunk_elems ? remaining : chunk_elems);

                float* chunk_buf = (float*)malloc(current_chunk * sizeof(float));
                if (!chunk_buf) {
                    #pragma omp critical
                    {
                        if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                            fprintf(stderr, "Failed to allocate chunk buffer for %s (%s)\n", txt_fp, strerror(errno));
                            atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                        }
                    }
                    break;
                }

                size_t read = fread(chunk_buf, sizeof(float), current_chunk, bin);
                if (read != current_chunk) {
                    #pragma omp critical
                    {
                        if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                            fprintf(stderr, "Failed to read chunk from %s (%s)\n", bin_fp, strerror(errno));
                            atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                        }
                    }
                    free(chunk_buf);
                    break;
                }

                uint64_t chunk_start_index = processed;
                size_t task_chunk_index = chunk_index;

                #pragma omp task firstprivate(chunk_buf, current_chunk, chunk_start_index, task_chunk_index) shared(results, error_flag, w, total_elements, txt_fp)
                {
                    int skip_task = atomic_load_explicit(&error_flag, memory_order_relaxed);
                    size_t capacity = 0;
                    char* text_buf = NULL;

                    if (!skip_task) {
                        size_t multiplier = max_chars_per_value + 2;
                        if (current_chunk > SIZE_MAX / multiplier) {
                            #pragma omp critical
                            {
                                if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                                    fprintf(stderr, "Chunk too large while converting %s\n", bin_fp);
                                    atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                                }
                            }
                            skip_task = 1;
                        } else {
                            capacity = current_chunk * multiplier;
                            if (capacity == 0) {
                                capacity = multiplier;
                            }
                            text_buf = (char*)malloc(capacity);
                            if (!text_buf) {
                                #pragma omp critical
                                {
                                    if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                                        fprintf(stderr, "Failed to allocate text chunk buffer for %s (%s)\n", txt_fp, strerror(errno));
                                        atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                                    }
                                }
                                skip_task = 1;
                            }
                        }
                    }

                    size_t written_chars = 0;
                    if (!skip_task && text_buf) {
                        int local_error = 0;
                        for (size_t i = 0; i < current_chunk; ++i) {
                            uint64_t global_index = chunk_start_index + i;
                            int written = snprintf(text_buf + written_chars,
                                                   capacity - written_chars,
                                                   "%.3f",
                                                   chunk_buf[i]);
                            if (written < 0) {
                                local_error = 1;
                                break;
                            }

                            size_t produced = (size_t)written;
                            if (produced >= capacity - written_chars) {
                                local_error = 1;
                                break;
                            }
                            written_chars += produced;

                            int end_of_row = ((global_index + 1) % w) == 0;
                            if (end_of_row) {
                                if (global_index + 1 != total_elements) {
                                    if (written_chars + 1 >= capacity) {
                                        local_error = 1;
                                        break;
                                    }
                                    text_buf[written_chars++] = '\n';
                                }
                            } else {
                                if (written_chars + 1 >= capacity) {
                                    local_error = 1;
                                    break;
                                }
                                text_buf[written_chars++] = ' ';
                            }
                        }

                        if (local_error) {
                            #pragma omp critical
                            {
                                if (!atomic_load_explicit(&error_flag, memory_order_relaxed)) {
                                    fprintf(stderr, "Failed to format chunk while converting %s\n", bin_fp);
                                    atomic_store_explicit(&error_flag, 1, memory_order_relaxed);
                                }
                            }
                            skip_task = 1;
                        }
                    }

                    if (!skip_task && text_buf) {
                        results[task_chunk_index].data = text_buf;
                        results[task_chunk_index].length = written_chars;
                    } else {
                        free(text_buf);
                    }
                    free(chunk_buf);
                }

                processed += current_chunk;
                chunk_index++;
            }

            #pragma omp taskwait
        }
    }

    fclose(bin);

    if (atomic_load_explicit(&error_flag, memory_order_relaxed)) {
        for (size_t i = 0; i < chunk_count; ++i) {
            free(results[i].data);
        }
        free(results);
        fclose(txt);
        remove(txt_fp);
        return;
    }

    for (size_t i = 0; i < chunk_count; ++i) {
        if (results[i].data && results[i].length > 0) {
            if (fwrite(results[i].data, sizeof(char), results[i].length, txt) != results[i].length) {
                fprintf(stderr, "Failed to write formatted chunk to %s (%s)\n", txt_fp, strerror(errno));
                for (size_t j = 0; j < chunk_count; ++j) {
                    free(results[j].data);
                }
                free(results);
                fclose(txt);
                remove(txt_fp);
                return;
            }
        }
        free(results[i].data);
    }

    free(results);
    fclose(txt);
}

void get_dimension_txt(FILE* matrix_file, uint32_t* h, uint32_t* w) {
    if (fscanf(matrix_file, "%d %d", h, w) != 2) {fprintf(stderr, "Input Matrix has an invalid dimension header"); return; }
}
