NVCC = nvcc
NVCCFLAGS = -O2

TARGET = histogram
SRC = histogram.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
