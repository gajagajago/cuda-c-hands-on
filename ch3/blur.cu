#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "encodeDecode.h"

#define CHANNEL 4
#define BLURSIZE 1

__global__
void blurKernel(unsigned char* in, unsigned char* out, int w, int h)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        int pixels = 0;
        int aggrPixelVal = 0;

        for (int r = Row-BLURSIZE; r <= Row+BLURSIZE; r++) {
            for (int c = Col-BLURSIZE; c <= Col+BLURSIZE; c++) {
                int offset = r * w + c;

                if (-1 < r && r < h && -1 < c && c < w) {
                    pixels ++;
                    aggrPixelVal += (int)in[offset];
                }
            }
        }

        int outOffset = Row * w + Col;
        int outPixelVal = (unsigned char)(aggrPixelVal / pixels);

        out[outOffset * CHANNEL] = outPixelVal;
        out[outOffset * CHANNEL + 1] = outPixelVal;
        out[outOffset * CHANNEL + 2] = outPixelVal;
        out[outOffset * CHANNEL + 3] = 255;
    }
}

void blur(unsigned char* h_in, unsigned char* h_out, int m, int n) 
{
    int size = m * n * sizeof(unsigned char) * CHANNEL;

    unsigned char *d_in, *d_out;

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    blurKernel<<<dimGrid, dimBlock>>>(d_in, d_out, m, n);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: ./<executable_file> <image_file.png>\n");
        exit(1);    
    }

    char* filename = argv[1];
    
    int m = 512;
    int n = 512;

    unsigned char *h_in, *h_out;

    h_in = (unsigned char*)malloc(m * n * sizeof(unsigned char));
    h_out = (unsigned char*)malloc(m * n * sizeof(unsigned char) * CHANNEL);

    h_in = decode(filename);

    blur(h_in, h_out, m, n);

    encode("blurred_image.png", h_out, m, n);

    free(h_in);
    free(h_out);

    return 0;
}
