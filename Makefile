NVCC = nvcc
FLAGS = -O2 -arch=sm_75

all: benchmark_gpu benchmark_pesado

benchmark_gpu: benchmark_gpu.cu
	$(NVCC) $(FLAGS) benchmark_gpu.cu -o benchmark_gpu

benchmark_pesado: benchmark_pesado.cu
	$(NVCC) $(FLAGS) benchmark_pesado.cu -o benchmark_pesado

clean:
	rm -f benchmark_gpu benchmark_pesado
