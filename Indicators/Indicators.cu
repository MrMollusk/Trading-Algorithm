//CUDA lib streams
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <vector>
#include "indicatorsHEADER.h"
#include "API.h"
#include <math.h>

std::vector<float> jsonNumberExtractor(const std::vector<float>& vector, const int size) {
    std::vector<float> changedVector(vector.begin(), vector.begin() + std::min(size, (int)vector.size()));
    return changedVector;
}

float SMAKernelLaunch(const std::vector<float>& h_vector, int vectorElementNumber) {
    float* d_vector, * d_sum;
    float h_sum = 0;

    int vectorSize = sizeof(float) * vectorElementNumber;
    int threadsPerBlock = 32;
    int blocks = (vectorElementNumber + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&d_vector, vectorSize);
    cudaMalloc((void**)&d_sum, sizeof(float));

    cudaMemset(d_sum, 0, sizeof(float));

    cudaMemcpy(d_vector, h_vector.data(), vectorSize, cudaMemcpyHostToDevice);

    closeSum << <blocks, threadsPerBlock >> > (d_vector, d_sum, vectorElementNumber);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vector);
    cudaFree(d_sum);

    return h_sum / vectorElementNumber;
}

float EMAKernelLaunch(const std::vector<float>& h_vector, int ema) {
    const size_t vectorElementNumber = h_vector.size();
    float* d_vector, * d_ema;
    float smoothingFactor = 2.0 / (ema + 1);
    float initialEMA = 0.0;

    size_t vectorSize = sizeof(float) * vectorElementNumber;

    cudaMalloc((void**)&d_vector, vectorSize);
    cudaMalloc((void**)&d_ema, vectorSize);

    cudaMemcpy(d_vector, h_vector.data(), vectorSize, cudaMemcpyHostToDevice);

    for (int i = 0; i < ema; ++i) {
        initialEMA += h_vector[i];
    }
    initialEMA /= ema;

    cudaMemcpy(&d_ema[ema - 1], &initialEMA, sizeof(float), cudaMemcpyHostToDevice);

    EMA80Calculation<<<1, 1 >>> (d_vector, d_ema, smoothingFactor, ema, vectorElementNumber);
    cudaDeviceSynchronize();

    std::vector<float> h_ema(vectorElementNumber);
    cudaMemcpy(h_ema.data(), d_ema, vectorSize, cudaMemcpyDeviceToHost);

    cudaFree(d_vector);
    cudaFree(d_ema);

    return h_ema[vectorElementNumber - 1];
}


std::vector<float> stochasticCalculation(float* h_high, float* h_low, float* h_close, int stockNumber, int timeStep, int backPeriod, int SMAPeriod) {
    float* d_high, * d_low, * d_close, * d_K, * d_D;

    cudaMalloc((void**)&d_high, stockNumber * timeStep * sizeof(float));
    cudaMalloc((void**)&d_low, stockNumber * timeStep * sizeof(float));
    cudaMalloc((void**)&d_close, stockNumber * timeStep * sizeof(float));
    cudaMalloc((void**)&d_K, stockNumber * timeStep * sizeof(float));
    cudaMalloc((void**)&d_D, stockNumber * timeStep * sizeof(float));

    cudaMemcpy(d_high, h_high, stockNumber * timeStep * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_low, h_low, stockNumber * timeStep * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_close, h_close, stockNumber * timeStep * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (stockNumber + blockSize - 1) / blockSize;
    size_t sharedMemSize = 2 * backPeriod * sizeof(float);

    stochasticKernel <<<gridSize, blockSize, sharedMemSize >>> (d_high, d_low, d_close, d_K, d_D, backPeriod, SMAPeriod, stockNumber, timeStep);
    cudaDeviceSynchronize();

    std::vector<float> h_D(stockNumber * timeStep);
    cudaMemcpy(h_D.data(), d_D, stockNumber * timeStep * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_high);
    cudaFree(d_low);
    cudaFree(d_close);
    cudaFree(d_K);
    cudaFree(d_D);

    return h_D;
}

//Kernel for sum of close values
__global__ void closeSum(float* d_vector, float* d_sum, const int size) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if (idx < size) {
        atomicAdd(d_sum, d_vector[idx]);
    }
}

//Kernel to calculate 80EMA
__global__ void EMA80Calculation(float* d_vector, float* d_ema, float smoothingFactor, int ema, const size_t size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (size_t i = ema; i < size; ++i) {
            d_ema[i] = (d_vector[i] - d_ema[i - 1]) * smoothingFactor + d_ema[i - 1];
        }
    }
}

//Kernel for Stochastic Oscillator
__global__ void stochasticKernel(float* d_high, float* d_low, float* d_close, float* d_K, float* d_D, int backPeriod, int SMAPeriod, int stockNumber, int timeStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stockNumber) return;

    extern __shared__ float sharedData[];
    float* sharedHighs = sharedData;
    float* sharedLows = &sharedHighs[backPeriod];

    for (int t = 0; t < timeStep; ++t) {
        if (threadIdx.x < backPeriod && t >= threadIdx.x) {
            sharedHighs[threadIdx.x] = d_high[idx * timeStep + t - threadIdx.x];
            sharedLows[threadIdx.x] = d_low[idx * timeStep + t - threadIdx.x];
        }
        else if (threadIdx.x < backPeriod) {
            sharedHighs[threadIdx.x] = 0.0;
            sharedLows[threadIdx.x] = 0.0;
        }
        __syncthreads();

        float highHigh = sharedHighs[0];
        float lowLow = sharedLows[0];
        for (int i = 1; i < backPeriod; ++i) {
            highHigh = fmax(highHigh, sharedHighs[i]);
            lowLow = fmin(lowLow, sharedLows[i]);
        }

        float currentClose = d_close[idx * timeStep + t];
        if (highHigh != lowLow) {
            d_K[idx * timeStep + t] = ((currentClose - lowLow) / (highHigh - lowLow)) * 100.0;
        }
        else {
            d_K[idx * timeStep + t] = 0.0;
        }

        __syncthreads();

        if (t >= SMAPeriod - 1) {
            float sumK = 0.0;
            for (int i = 0; i < SMAPeriod; ++i) {
                sumK += d_K[idx * timeStep + t - i];
            }
            d_D[idx * timeStep + t] = sumK / SMAPeriod;
        }
    }
}
