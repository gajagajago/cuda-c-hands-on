#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 64 // Must be a multiple of 2
#define BLOCK_SIZE 32

__global__
void partialSumKernel(float* d_X)
{
    __shared__ float partialSum[BLOCK_SIZE];
    unsigned int tx = threadIdx.x;

    partialSum[tx] = d_X[blockDim.x * blockIdx.x + threadIdx.x];

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride >> 1) {
        __syncthreads();
        if (tx < stride) {
            partialSum[tx] += partialSum[tx + stride];
        }
    }

    if (tx == 0) {
        d_X[blockDim.x * blockIdx.x + threadIdx.x] = partialSum[tx];
    }
}

float partialSum(float* h_X)
{
    float* d_X;
    float total_sum = 0.0;

    int size = SIZE * sizeof(float);

    cudaMalloc((void**)&d_X, size);

    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(SIZE/BLOCK_SIZE, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    partialSumKernel<<<dimGrid, dimBlock>>>(d_X);

    // Copy only d_X[0]
    cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);

    cudaFree(d_X);

    for (int i = 0; i < SIZE; i = i + BLOCK_SIZE) {
        total_sum += h_X[i];
    }

    return total_sum;
}

int main(int argc, char* argv[])
{
    float* h_X;
    unsigned int size = SIZE * sizeof(float);

    h_X = (float*)malloc(size);

    for (unsigned int i = 0; i < SIZE; i++) {
        h_X[i] = (float)i;
    }

    float total_sum = partialSum(h_X);

    printf("Total sum: %.2f\n", total_sum);

    free(h_X);
}