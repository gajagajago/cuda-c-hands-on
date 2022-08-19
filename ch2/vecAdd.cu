#include <cuda.h>
#include <stdio.h>

#define threads_per_block 256.0

// n: vector size
__global__
void vecAddKernel(float* d_A, float* d_B, float* d_C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

// n: vector size
void vecAdd(float* A, float* B, float* C, int n)
{
    // 1. Init device variables
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // 2. Compute
    dim3 dimGrid(ceil(n/threads_per_block), 1, 1);
    dim3 dimBlock(threads_per_block, 1, 1);
    vecAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    // 3. Device -> Host & Free device memory
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char* argv[])
{   
    int n = atoi(argv[1]);

    float A[n], B[n], C[n];

    for (int i=0; i<n; i++) {
        A[i] = B[i] = i;
    }

    vecAdd(A, B, C, n);

    for (int i=0; i<n; i++) {
        printf("%f\t", C[i]);
    }
    printf("\n");

    return 0;
}