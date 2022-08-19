#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

// C[i][j] = A[i][j] + B[i][j]
__global__
void matrixAddKernel(float* C, float* A, float* B, int n)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Row < n && Col < n) {
        int offset = Row * n + Col;
        C[offset] = A[offset] + B[offset];
    }
}

// h_C, h_A, h_B are linearized 2D arrays
// Assumne all 2D arrays are of n * n size
void matrixAdd(float* h_C, float* h_A, float* h_B, int n)
{
    int size = n * n * sizeof(float);

    // Device variables
    float *d_C, *d_A, *d_B;

    // Allocate device address space 
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    // Memory copy host input arrays 
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Call kernel
    dim3 dimGrid(ceil(n/16.0), ceil(n/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    matrixAddKernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, n);

    // Memory copy device output array
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: ./<executable_file> <dimension size of 2D array>\n");
        exit(1);
    }

    int n = atoi(argv[1]);
    int rows = n;
    int cols = n;

    int size = n * n * sizeof(float);

    float *h_C, *h_A, *h_B;

    h_C = (float*)malloc(size); 
    h_A = (float*)malloc(size); 
    h_B = (float*)malloc(size); 

    matrixAdd(h_C, h_A, h_B, n);

    for (int i=0; i < rows; i++) {
        for (int j=0; j <cols; j++) {
            printf("%.2f ", h_C[i*rows + j]);
        }
        printf("\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);
}