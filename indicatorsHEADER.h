#ifndef INDICATORS_KERNELS_H
#define INDICATORS_KERNELS_H

#include <cuda_runtime.h>
#include <vector>

std::vector<float> jsonNumberExtractor(const std::vector<float>& vector, const int size);

float SMAKernelLaunch(const std::vector<float>& h_vector, int vectorElementNumber);

float EMAKernelLaunch(const std::vector<float>& h_vector, int ema);

std::vector<float> stochasticCalculation(float* h_high, float* h_low, float* h_close, int stockNumber, int timeStep, int backPeriod, int SMAPeriod);

__global__ void closeSum(float* d_vector, float* d_sum, const int size);

__global__ void EMA80Calculation(float* d_vector, float* d_ema, float smoothingFactor, int ema, const size_t size);

__global__ void stochasticKernel(float* d_high, float* d_low, float* d_close, float* d_K, float* d_D, int backPeriod, int SMAPeriod, int stockNumber, int timeStep);

#endif
