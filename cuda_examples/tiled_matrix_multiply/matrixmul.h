#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_


// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

#define TILE_SIZE 32
#define MATRIX_SIZE 1024

#endif // _MATRIXMUL_H_

