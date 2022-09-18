#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define MASK_WIDTH 3
#define TILE_WIDTH 2

float mask[MASK_WIDTH][MASK_WIDTH] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1}
};
__constant__ float M[MASK_WIDTH][MASK_WIDTH];

__global__
void tiledConvolution2DKernel(float* d_N, float* d_P, int width)
{
    int out_row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int out_col_idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH];

    int conv_row_start_idx = out_row_idx - MASK_WIDTH / 2;
    int conv_col_start_idx = out_col_idx - MASK_WIDTH / 2;

    // 1. Load to shared memory
    N_ds[threadIdx.y][threadIdx.x] = d_N[out_row_idx * width + out_col_idx];

    __syncthreads();

    // 2. Convolution
    float p = 0.0;
    int conv_row_ptr, conv_col_ptr;

    for (int i = 0; i < MASK_WIDTH; i++) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            conv_row_ptr = conv_row_start_idx + i;
            conv_col_ptr = conv_col_start_idx + j;

            int conv_row_valid = conv_row_ptr >= 0 && conv_row_ptr < width;
            int conv_col_valid = conv_col_ptr >= 0 && conv_col_ptr < width;

            if (conv_row_valid && conv_col_valid) {
                int conv_row_in_Nds = conv_row_ptr >= blockDim.y * blockIdx.y && conv_row_ptr < blockDim.y * (blockIdx.y + 1);
                int conv_col_in_Nds = conv_col_ptr >= blockDim.x * blockIdx.x && conv_col_ptr < blockDim.x * (blockIdx.x + 1);

                if (conv_row_in_Nds && conv_col_in_Nds) {
                    p += N_ds[conv_row_ptr - blockDim.y * blockIdx.y][conv_col_ptr - blockDim.x * blockIdx.x] * M[i][j]; // This is potential problem
                } else {
                    p += d_N[conv_row_ptr * width + conv_col_ptr] * M[i][j];
                }
            }
        }
    }

    d_P[out_row_idx * width + out_col_idx] = p;
}

void tiledConvolution2D(float* N, float* P, int width)
{
    float *d_N, *d_P;

    // 1. Init mask constant array
    cudaMemcpyToSymbol(M, mask, MASK_WIDTH * MASK_WIDTH * sizeof(float), 0, cudaMemcpyHostToDevice);

    // 2. Init device arrays
    cudaMalloc((void**)&d_N, width * width * sizeof(float));
    cudaMalloc((void**)&d_P, width * width * sizeof(float));

    cudaMemcpy(d_N, N, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Init execution configuration parameters
    dim3 dimGrid(ceil(float(width) / TILE_WIDTH), ceil(float(width) / TILE_WIDTH), 1);
    dim3 dimBlock((float)TILE_WIDTH, (float)TILE_WIDTH, 1);

    // 4. Launch kernel
    tiledConvolution2DKernel<<<dimGrid, dimBlock>>>(d_N, d_P, width);

    // 5. Retrieve result
    cudaMemcpy(P, d_P, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Free kernel data structures
    cudaFree(d_N);
    cudaFree(d_P);
}

int main(int argc, char* argv[]) 
{
    if (argc != 2) {
        printf("Usage: ./<executable_file> <dimension size of 2D array>\n");
        exit(1);
    }

    int width = atoi(argv[1]);

    float inArr[width][width], outArr[width][width];

    printf("[inArr]\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            inArr[i][j] = i * width + j;
            printf("%.1f ", inArr[i][j]);
        }
        printf("\n");
    }

    tiledConvolution2D(&inArr[0][0], &outArr[0][0], width);

    printf("[outArr]\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.1f ", outArr[i][j]);
        }
        printf("\n");
    }
}