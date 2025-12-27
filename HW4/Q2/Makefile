NVCC = nvcc
NVCCFLAGS = -O3 --use_fast_math -arch=sm_120
LDFLAGS = -lcublas -lcusparse
 
TARGET = heat_eq
SRC = heat_eq.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
