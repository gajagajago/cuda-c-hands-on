#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 101
#define BLOCK_SIZE 4

// 1d array partial sum 
// 1 thread per 1 element
__global__
void partialSumKernel(float* d_X)
{
    __shared__ float partialSum[BLOCK_SIZE];

    int h_Idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tx = threadIdx.x;

    // Boundary condition
    if (h_Idx < SIZE) {
        partialSum[tx] = d_X[h_Idx];
    } else {
        partialSum[tx] = 0.0;
    }

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride >> 1) {
        __syncthreads();
        if (tx < stride) {
            partialSum[tx] += partialSum[tx + stride];
        }
    }

    if (tx == 0) {
        d_X[h_Idx] = partialSum[tx];
    }
}

// 1d array partial sum optimized
// 1 thread per 2 elements
// Execution configuration parameters: dimBlock should be BLOCKSIZE / 2
__global__
void partialSumOptKernel(float* d_X)
{
    __shared__ float partialSum[BLOCK_SIZE];

    int h_Idx = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tx = threadIdx.x;
    unsigned int stride = blockDim.x;

    // Boundary condition
    if (h_Idx < SIZE) {
        partialSum[tx] = d_X[h_Idx];
        partialSum[tx + stride] = d_X[h_Idx + stride];
    } else {
        partialSum[tx] = 0.0;
    }

    for (; stride >= 1; stride = stride >> 1) {
        __syncthreads();

        // For the first iteration, no control divergence
        if (tx < stride) {
            partialSum[tx] += partialSum[tx + stride];
        }
    }

    if (tx == 0) {
        d_X[h_Idx] = partialSum[tx];
    }
}

float partialSum(float* h_X)
{
    float* d_X;
    float total_sum = 0.0;

    int size = SIZE * sizeof(float);

    cudaMalloc((void**)&d_X, size);

    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);

    // dim3 dimGrid(ceil((float)SIZE/BLOCK_SIZE), 1, 1);
    // dim3 dimBlock(BLOCK_SIZE, 1, 1);
    // partialSumKernel<<<dimGrid, dimBlock>>>(d_X);

    dim3 dimGrid(ceil((float)SIZE/BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE/2, 1, 1); 
    partialSumOptKernel<<<dimGrid, dimBlock>>>(d_X);

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