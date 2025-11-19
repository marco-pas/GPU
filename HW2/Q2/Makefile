NVCC = nvcc
NVCCFLAGS = -O2

TARGET = reduction
SRC = reduction.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
