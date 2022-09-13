#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define HORIZONTAL_BLOCK_GRANULARITY 2 // Each thread block computes $HORIZONTAL_BLOCK_GRANULARITY tiles adjacent in horizontal direction
#define TILE_WIDTH 2

// TODO: Add boundary checks, especially for granularity existence 
__global__
void matMulOptKernel(float* d_M, float* d_N, float* d_P, int width) 
{
    // 1. Init shared memory data structures
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][HORIZONTAL_BLOCK_GRANULARITY * TILE_WIDTH];

    // 2. Init local accumulated variables
    float p_vals[HORIZONTAL_BLOCK_GRANULARITY] = {0.0};

    // 3. Find position(s) in d_P
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Cols[HORIZONTAL_BLOCK_GRANULARITY];

    for (int i = 0; i < HORIZONTAL_BLOCK_GRANULARITY; i++) {
        Cols[i] = blockDim.x * (HORIZONTAL_BLOCK_GRANULARITY * blockIdx.x + i) + threadIdx.x;
    }

    // 4. Phase by phase calculation
    for (int phase = 0; phase < ceil((float)width / TILE_WIDTH); phase++) {
        // 4-1. Load elements to shared memory DS
        // Load 1 element for Mds, HORIZONTAL_BLOCK_GRANULARITY for Nds
        int d_M_Col = phase * TILE_WIDTH + threadIdx.x;
        int d_N_Row = phase * TILE_WIDTH + threadIdx.y;
        
        Mds[threadIdx.y][threadIdx.x] = d_M[Row * width + d_M_Col];
        for (int i = 0; i < HORIZONTAL_BLOCK_GRANULARITY; i++) {
            // Made mistake once here: Nds[...]["i" * ...] not "phase"
            Nds[threadIdx.y][i * TILE_WIDTH + threadIdx.x] = d_N[d_N_Row * width + Cols[i]];
        }

        __syncthreads();

        // 4-2. Compute
        for (int j = 0; j < TILE_WIDTH; j++) {
            for (int i = 0; i < HORIZONTAL_BLOCK_GRANULARITY; i++) {
                p_vals[i] += Mds[threadIdx.y][j] * Nds[j][threadIdx.x + TILE_WIDTH * i];
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < HORIZONTAL_BLOCK_GRANULARITY; i++) {
        d_P[Row * width + Cols[i]] = p_vals[i];
    }
}

void matMul(float* h_M, float* h_N, float* h_P, int width)
{
    int size = width * width * sizeof(float);

    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float)width/(TILE_WIDTH * HORIZONTAL_BLOCK_GRANULARITY)), ceil((float)width/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    matMulOptKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);

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

    for (int i=0; i<width; i++) {
        for (int j=0; j<width; j++) {
            // // Random initialize two input arrays
            // int offset = i*width + j;
            // srand(time(NULL));
            // h_M[offset] = (float)(rand()/RAND_MAX);
            // srand(time(NULL));
            // h_N[offset] = (float)(rand()/RAND_MAX);
            h_M[i * width + j] = (float)(i + j);
            h_N[i * width + j] = (float)(i + j);
        }
    }

    matMul(h_M, h_N, h_P, width);

    for (int i=0; i<width; i++) {
        for (int j=0; j<width; j++) {
            printf("%.1f ", h_P[i * width + j]);
        }
        printf("\n");
    }
}