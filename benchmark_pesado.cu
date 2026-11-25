#include <cstdio>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Conta pesada na GPU (500 iterações matemáticas por elemento)
__global__ void calculo_gpu(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = a[i];
        float mult = b[i];
        for (int k = 0; k < 500; ++k) {
            val = sinf(val) * cosf(mult) + sqrtf(fabsf(val) + 1.0f);
        }
        c[i] = val;
    }
}

// Exatamente a mesma conta pesada na CPU
void calculo_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        float val = a[i];
        float mult = b[i];
        for (int k = 0; k < 500; ++k) {
            val = std::sin(val) * std::cos(mult) + std::sqrt(std::fabs(val) + 1.0f);
        }
        c[i] = val;
    }
}

int main() {
    const int n = 2000000; // 2 milhões de elementos
    const size_t bytes = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c_cpu = new float[n];
    float *h_c_gpu = new float[n];

    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.5f;
        h_b[i] = 2.5f;
    }

    // 1. Medição na CPU
    auto inicio_cpu = std::chrono::high_resolution_clock::now();
    calculo_cpu(h_a, h_b, h_c_cpu, n);
    auto fim_cpu = std::chrono::high_resolution_clock::now();
    double tempo_cpu_ms = std::chrono::duration<double, std::milli>(fim_cpu - inicio_cpu).count();

    // 2. Preparação da GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // WARM-UP: acorda a GPU antes de ligar o cronômetro!
    calculo_gpu<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 3. Medição real na GPU
    cudaEventRecord(start);
    calculo_gpu<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tempo_gpu_ms = 0.0f;
    cudaEventElapsedTime(&tempo_gpu_ms, start, stop);

    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Conferencia (CPU vs GPU): %.4f vs %.4f\n", h_c_cpu[0], h_c_gpu[0]);
    printf("Tempo CPU (i7-11800H)   : %.3f ms\n", tempo_cpu_ms);
    printf("Tempo GPU (GTX 1650)    : %.3f ms\n", tempo_gpu_ms);
    printf("Speedup (GPU vs CPU)    : %.1fx mais rapido!\n", tempo_cpu_ms / tempo_gpu_ms);

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
