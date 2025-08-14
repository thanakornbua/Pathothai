#include "loader_common.h"
#include <string.h>

int extract_channel(const unsigned char *src, int w, int h, int src_channels, int channel_index, unsigned char *dst) {
    if (!src || !dst || channel_index < 0 || channel_index >= src_channels) return -1;
    if (src_channels == 1) { // grayscale copy
        memcpy(dst, src, (size_t)w*h);
        return 0;
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const unsigned char *p = src + (y*w + x)*src_channels;
            dst[y*w + x] = p[channel_index];
        }
    }
    return 0;
}
