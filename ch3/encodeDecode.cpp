#include <stdio.h>
#include "lodepng.h"

// Encode from raw pixels to disk
void encode(const char* filename, unsigned char* image, int width, int height) 
{
    unsigned error = lodepng_encode32_file(filename, image, (unsigned)width, (unsigned)height);

    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

// Decode from disk to raw pixels
unsigned char* decode(const char* filename)
{
    unsigned char* image;
    unsigned width, height;

    unsigned error = lodepng_decode32_file(&image, &width, &height, filename);
    
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

    return image;
}