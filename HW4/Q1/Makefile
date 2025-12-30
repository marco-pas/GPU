NVCC = nvcc
NVCCFLAGS = -O3 --use_fast_math -arch=sm_120
 
TARGET = convolution
SRC = convolution.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
