NVCC = nvcc
FLAGS = -arch=sm_70 -O3

all: matrix matrix_timeloop matrix_timeloop_double

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f vectorAdd vectorAdd_timeloop

.PHONY: all clean
