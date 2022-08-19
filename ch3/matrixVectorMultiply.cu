#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// outVec[j] = inMatrix[j][i] dot inVec[i]
// Assume inMatrix size is n * n, Vector size is n
__global__ 
void matrixVectorMultiplyKernel(float* outVec, float* inMatrix, float* inVec, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < n) {
        float aggr = 0.0;

        for (int i=0; i < n; i++) {
            aggr += inMatrix[j * n + i] * inVec[i];
        }

        outVec[j] = aggr;
    }
}

void matrixVectorMultiply(float* h_outVec, float* h_inMatrix, float* h_inVec, int n)
{
    int matrixSize = n * n * sizeof(float);
    int vectorSize = n * sizeof(float);

    float *d_outVec, *d_inMatrix, *d_inVec;

    cudaMalloc((void**)&d_outVec, vectorSize);
    cudaMalloc((void**)&d_inMatrix, matrixSize);
    cudaMalloc((void**)&d_inVec, vectorSize);

    cudaMemcpy(d_inMatrix, h_inMatrix, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inVec, h_inVec, vectorSize, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(n/4.0), 1, 1);
    dim3 dimBlock(4.0, 1, 1);
    matrixVectorMultiplyKernel<<<dimGrid, dimBlock>>>(d_outVec, d_inMatrix, d_inVec, n);

    cudaMemcpy(h_outVec, d_outVec, vectorSize, cudaMemcpyDeviceToHost);

    cudaFree(d_outVec);
    cudaFree(d_inMatrix);
    cudaFree(d_inVec);
}

int main(int argc, char* argv[]) 
{
    if (argc != 2) {
        printf("Usage: ./<executable_file> <dimension size>\n");
        exit(1);
    }

    int n = atoi(argv[1]);

    int matrixSize = n * n * sizeof(float);
    int vectorSize = n * sizeof(float);    

    float *h_outVec, *h_inMatrix, *h_inVec;

    h_outVec = (float*)malloc(vectorSize);
    h_inMatrix = (float*)malloc(matrixSize);
    h_inVec = (float*)malloc(vectorSize);

    // Init inputs
    for (int j=0; j<n; j++) {
        for (int i=0; i<n; i++) {
            h_inMatrix[j*n + i] = j*n + i;
        }
        h_inVec[j] = j;
    }

    matrixVectorMultiply(h_outVec, h_inMatrix, h_inVec, n);

    for (int j=0; j<n; j++) {
        printf("%.1f ", h_outVec[j]);
    }
    printf("\n");

    free(h_outVec);
    free(h_inMatrix);
    free(h_inVec);
}
