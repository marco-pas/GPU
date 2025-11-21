NVCC = nvcc
NVCCFLAGS = -O2

TARGET = matrix_mul
SRC = matrix_mul.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
