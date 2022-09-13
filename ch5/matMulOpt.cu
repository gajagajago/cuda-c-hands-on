#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define HORIZONTAL_BLOCK_GRANULARITY 2 // Each thread block computes $HORIZONTAL_BLOCK_GRANULARITY tiles adjacent in horizontal direction
#define TILE_WIDTH 4

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
        Cols[i] = blockDim.x * (blockIdx.x + i) + threadIdx.x;
    }

    // 4. Phase by phase calculation
    for (int phase = 0; phase < ceil((float)width / TILE_WIDTH); phase++) {
        // 4-1. Load elements to shared memory DS
        // Load 1 element for Mds, HORIZONTAL_BLOCK_GRANULARITY for Nds
        int d_M_Col = phase * TILE_WIDTH + threadIdx.x;
        int d_N_Row = phase * TILE_WIDTH + threadIdx.y;
        
        Mds[threadIdx.y][threadIdx.x] = d_M[Row * width + d_M_Col];
        for (int i = 0; i < HORIZONTAL_BLOCK_GRANULARITY; i++) {
            Nds[threadIdx.y][phase * TILE_WIDTH + threadIdx.x] = d_N[d_N_Row * width + Cols[i]];
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