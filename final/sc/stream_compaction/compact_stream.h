#ifndef _compact_stream_H_
#define _compact_stream_H_


// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

#define TILE_SIZE 32

#endif // compact_stream

