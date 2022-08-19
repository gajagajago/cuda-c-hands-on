#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "encodeDecode.h"

#define CHANNEL 4

__global__
void colorToGrayscaleConversionKernel(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < width && Row < height) {
        int grayOffset = Row * width + Col;

        int rgbOffset = CHANNEL * grayOffset;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[rgbOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
        Pout[rgbOffset+1] = 0.21f*r + 0.71f*g + 0.07f*b;
        Pout[rgbOffset+2] = 0.21f*r + 0.71f*g + 0.07f*b;
        Pout[rgbOffset+3] = 255;
    }
}

void colorToGrayscaleConversion(unsigned char* h_Pout, unsigned char* h_Pin, int m, int n)
{
    int size = m * n * sizeof(unsigned char) * CHANNEL;
    unsigned char *d_Pin, *d_Pout;

    cudaMalloc((void**)&d_Pin, size);
    cudaMalloc((void**)&d_Pout, size);

    cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(m / 16.0), ceil(n / 16.0), 1);
    dim3 dimBlock(16, 16, 1);
    colorToGrayscaleConversionKernel<<<dimGrid, dimBlock>>>(d_Pout, d_Pin, m, n);

    cudaMemcpy(h_Pout, d_Pout, size, cudaMemcpyDeviceToHost);

    cudaFree(d_Pin);
    cudaFree(d_Pout);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: ./<executable_file> <image_file.png>\n");
        exit(1);    
    }

    const char* filename = argv[1];

    unsigned char *h_Pin, *h_Pout;

    int m = 512;
    int n = 512;

    h_Pin = (unsigned char*)malloc(m * n * sizeof(unsigned char));
    h_Pout = (unsigned char*)malloc(m * n * sizeof(unsigned char) * CHANNEL);

    printf("decoding image...\n");
    h_Pin = decode(filename);

    printf("colorToGrayscaleConversion...\n");
    colorToGrayscaleConversion(h_Pout, h_Pin, m, n);

    printf("encoding converted image...\n");
    encode("image_converted.png", h_Pout, m, n);

    printf("finish\n");
    free(h_Pin);
    free(h_Pout);

    return 0;
}