#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 4

// P = M * N
// Assumption
// 1. M, N is a square matrix of size width x width
// 2. width is a multiple of TILE_WIDTH
__global__ 
void tiledMatMulKernel(float* d_M, float* d_N, float* d_P, int width)
{
    float pval = 0.0; 

    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Tile data structure cached at shared memory under block scope
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    for (int phase=0; phase<ceil(width / (float)TILE_WIDTH); phase++) {
        // Load designated tile element
        // Collectively, each phase uses one tile of M and one tile of N
        int d_M_Col = phase * TILE_WIDTH + tx;
        int d_N_Row = phase * TILE_WIDTH + ty;

        if (Row < width && d_M_Col < width)
            Mds[ty][tx] = d_M[Row * width + d_M_Col];
        else 
            Mds[ty][tx] = 0.0;
        if (d_N_Row < width && Col < width)
            Nds[ty][tx] = d_N[d_N_Row * width + Col];
        else 
            Nds[ty][tx] = 0.0;

        __syncthreads();    // Synchronization - Tile elements of this phase are loaded

        for (int i=0; i<TILE_WIDTH; i++) {
            pval += Mds[ty][i] * Nds[i][tx];
        }

        __syncthreads();    // Synchronization - Wait all threads to finish this phase. Tile elements in shared memory will be refreshed.
    }

    if (Row < width && Col < width)
        d_P[Row * width + Col] = pval;
}

__global__
void vanillaMatMulKernel(float* d_M, float* d_N, float* d_P, int width)
{
    float pval = 0.0; 

    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < width && Row < width) {
        for (int k=0; k<width; k++) {
            pval += d_M[Row*width + k] * d_N[k*width + Col];
        }
    }

    d_P[Row * width + Col] = pval;
}

// Last parameter is the desired matmul kernel function
void matMul(float* h_M, float* h_N, float* h_P, int width, void (*matMulKernel)(float*, float*, float* ,int))
{
    int size = width * width * sizeof(float);

    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(width/TILE_WIDTH, width/TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    matMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: ./<executable_file> <dimension size of 2D array>\n");
        exit(1);
    }

    float *h_M, *h_N, *h_P;

    int width = atoi(argv[1]);
    int size = width * width * sizeof(float);

    h_M = (float*)malloc(size);
    h_N = (float*)malloc(size);
    h_P = (float*)malloc(size);

    // Random initialize two input arrays
    for (int i=0; i<width; i++) {
        for (int j=0; j<width; j++) {
            int offset = i*width + j;
            srand(time(NULL));
            h_M[offset] = (float)(rand()/RAND_MAX);
            srand(time(NULL));
            h_N[offset] = (float)(rand()/RAND_MAX);
        }
    }

    // Initialize clock variables
    clock_t s1, s2, e1, e2;

    s1 = clock();
    matMul(h_M, h_N, h_P, width, &tiledMatMulKernel);
    e1 = clock();

    s2 = clock();
    matMul(h_M, h_N, h_P, width, &vanillaMatMulKernel);
    e2 = clock();

    printf("[Elapsed Time]"
    "\tTiled: %.3f"
    "\tVanilla: %.3f"
    "\n", 
    (float)((e1-s1)/CLOCKS_PER_SEC), 
    (float)((e2-s2))/CLOCKS_PER_SEC);

    printf("[s1]%d [e1]%d [s2]%d [e2]%d\n", int(s1), int(e1), int(s2), int(e2));
}