#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

// Código paralelo na placa de vídeo
__global__ void soma_gpu(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Código sequencial no processador (CPU)
void soma_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 10000000; // 10 milhões de números
    const size_t bytes = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c_cpu = new float[n];
    float *h_c_gpu = new float[n];

    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.5f;
        h_b[i] = 2.5f;
    }

    // Medição na CPU
    auto inicio_cpu = std::chrono::high_resolution_clock::now();
    soma_cpu(h_a, h_b, h_c_cpu, n);
    auto fim_cpu = std::chrono::high_resolution_clock::now();
    double tempo_cpu_ms = std::chrono::duration<double, std::milli>(fim_cpu - inicio_cpu).count();

    // Alocação na VRAM da GTX 1650
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Medição na GPU
    cudaEventRecord(start);
    soma_gpu<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tempo_gpu_ms = 0.0f;
    cudaEventElapsedTime(&tempo_gpu_ms, start, stop);

    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Resultado conferido (esperado 4.0): %.1f\n", h_c_gpu[n - 1]);
    printf("Tempo CPU (i7-11800H): %.3f ms\n", tempo_cpu_ms);
    printf("Tempo GPU (GTX 1650) : %.3f ms\n", tempo_gpu_ms);
    printf("Speedup (GPU vs CPU) : %.1fx mais rapido!\n", tempo_cpu_ms / tempo_gpu_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;

    return 0;
}
