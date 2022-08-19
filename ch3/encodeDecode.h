#ifndef ENCODE_DECODE_H
#define ENCODE_DECODE_H

// Encode from raw pixels to disk
void encode(const char* filename, unsigned char* image, int width, int height);

// Decode from disk to raw pixels
unsigned char* decode(const char* filename);

#endif