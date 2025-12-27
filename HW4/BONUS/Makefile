NVCC = nvcc
NVCCFLAGS = -O3 --use_fast_math -arch=sm_120
 
TARGET = wmma_gemm
SRC = wmma_gemm.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
