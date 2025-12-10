NVCC       := nvcc
TARGET     := main
SRC        := main.cu
ARCH       := -gencode arch=compute_80,code=sm_80
CFLAGS     := -O3 $(ARCH)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
