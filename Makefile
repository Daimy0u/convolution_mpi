SETCC := export OMPI_CC=gcc-15;

MPICC := mpicc
CC := gcc
MAC_CC := gcc-15

CFLAGS := -std=c11 -O3 -march=native -Wno-unused-result -I./include -fopenmp -D_POSIX_C_SOURCE=200112L
LDLIBS := -lm

SRC := src/file.c src/generate.c src/matrix.c src/cli_parse.c \
		src/conv_openmp.c src/conv_mpi.c src/conv_utils.c

OUT := conv_stride

.PHONY: all clean

ifeq ($(shell uname -s),Darwin)
	mac: $(MAC)
else
	all: $(OUT)
endif


$(OUT): $(SRC) src/main.c
	$(MPICC) $(CFLAGS) $(SRC) src/main.c -o $(OUT) $(LDLIBS)

mac: $(SRC) src/main.c
	$(SETCC) $(MPICC) $(CFLAGS) $(SRC) src/main.c -o $(OUT) $(LDLIBS)


clean:
	-rm -f $(OUT)
