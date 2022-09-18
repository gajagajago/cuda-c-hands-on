#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define MASK_WIDTH 3
#define TILE_WIDTH 4

float mask[MASK_WIDTH] = {1, 2, 1};
__constant__ float M[MASK_WIDTH];

__global__
void tiledConvolution1DKernel(float* d_N, float* d_P, int width)
{
    __shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1];
    int n = MASK_WIDTH / 2;

    // 1. Load left halo cells
    int halo_idx_left = blockDim.x * (blockIdx.x - 1) + threadIdx.x;

    if (blockDim.x - threadIdx.x <= n) {
        N_ds[threadIdx.x - (blockDim.x - n)] = halo_idx_left >= 0 ? d_N[halo_idx_left] : 0;
    }

    // 2. Load right halo cells
    int halo_idx_right = blockDim.x * (blockIdx.x + 1) + threadIdx.x;

    if (threadIdx.x < n) {
        N_ds[n + blockDim.x + threadIdx.x] = halo_idx_right < width ? d_N[halo_idx_right] : 0;
    }

    // 3. Load internal cells
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    N_ds[n + threadIdx.x] = i < width ? d_N[i] : 0;

    // 4. sync
    __syncthreads();

    // 5. Convolution
    if (i < width) {

        float p = 0.0;

        for (int j = 0; j < MASK_WIDTH; j++) {
            p += N_ds[threadIdx.x + j] * M[j];
        }

        d_P[i] = p;
    }
}

void tiledConvolution1D(float* N, float* P, int width)
{   
    float *d_N, *d_P;

    // 1. Init mask constant array
    cudaMemcpyToSymbol(M, mask, sizeof(float) * MASK_WIDTH, 0, cudaMemcpyHostToDevice);

    // 2. Init device arrays
    cudaMalloc((void**)&d_N, sizeof(float) * width);
    cudaMalloc((void**)&d_P, sizeof(float) * width);

    cudaMemcpy(d_N, N, sizeof(float) * width, cudaMemcpyHostToDevice);

    // 3. Init execution configuration parameters
    dim3 dimGrid(ceil((float)width / TILE_WIDTH), 1, 1);
    dim3 dimBlock((float)TILE_WIDTH, 1, 1);

    // 4. Launch kernel
    tiledConvolution1DKernel<<<dimGrid, dimBlock>>>(d_N, d_P, width);

    // 5. Retrieve result
    cudaMemcpy(P, d_P, sizeof(float) * width, cudaMemcpyDeviceToHost);

    // 6. Free device arrays
    cudaFree(d_N);
    cudaFree(d_P);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: ./<executable_file> <dimension size of 1D array>\n");
        exit(1);
    }

    int width = atoi(argv[1]);

    float inArr[width], outArr[width];

    for (int i = 0; i < width; i++) {
        inArr[i] = i;
    }

    tiledConvolution1D(inArr, outArr, width);

    for (int i = 0; i < width; i++) {
        printf("%.1f ", inArr[i]);
    }
    printf("* ");

    for (int i = 0; i < MASK_WIDTH; i++) {
        printf("%.1f ", mask[i]);
    }
    printf("= ");

    for (int i = 0; i < width; i++) {
        printf("%.1f ", outArr[i]);
    }
    printf("\n");
}