#pragma once
#include <stdint.h>

typedef struct {
    int64_t width;
    int64_t height;
    int channels; // 1 or 3
    // accessor to fetch a region (x,y,w,h) into user buffer RGBA8 or RGB8
    // returns 0 on success
    int (*read_region)(void *ctx, int64_t x, int64_t y, int64_t w, int64_t h, int req_channels, unsigned char *out);
    void (*destroy)(void *ctx);
    void *ctx;
} ImageHandle;

// Utility to extract a single channel viewport (R/G/B) into output
int extract_channel(const unsigned char *src, int w, int h, int src_channels, int channel_index, unsigned char *dst);
