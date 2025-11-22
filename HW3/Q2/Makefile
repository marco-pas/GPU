NVCC = nvcc
NVCCFLAGS = -O2 -g -lineinfo

TARGET = vecAdd_streams
SRC = vecAdd_streams.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
