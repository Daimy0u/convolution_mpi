UNAME_S := $(shell uname)

MPICC := mpicc
CC := gcc

ifeq ($(UNAME_S),Darwin)
GCC_BREW := gcc-15
MPICC := OMPI_CC=$(GCC_BREW) mpicc
CC := $(GCC_BREW)
endif

CFLAGS := -std=c11 -O3 -march=native -Wno-unused-result -I./include -fopenmp -D_POSIX_C_SOURCE=200112L
LDLIBS := -lm

SRC := src/file.c src/generate.c src/matrix.c src/cli_parse.c \
		src/conv_openmp.c src/conv_mpi.c src/conv_utils.c

OUT := conv_stride

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) src/main.c
	$(MPICC) $(CFLAGS) $(SRC) src/main.c -o $(OUT) $(LDLIBS)

clean:
	-rm -f $(OUT)
