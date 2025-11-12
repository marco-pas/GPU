NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O3

TARGETS = vectorAdd vectorAdd_timeloop

all: $(TARGETS)

vectorAdd: vectorAdd.cu
	$(NVCC) $(NVCC_FLAGS) vectorAdd.cu -o vectorAdd

vectorAdd_timeloop: vectorAdd_timeloop.cu
	$(NVCC) $(NVCC_FLAGS) vectorAdd_timeloop.cu -o vectorAdd_timeloop

clean:
	rm -f $(TARGETS)

.PHONY: all clean

