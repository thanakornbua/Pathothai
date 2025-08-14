#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "loader_common.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char *pixels; // RGB8
    int w, h, c;
} RasterCtx;

static int raster_read_region(void *vctx, int64_t x, int64_t y, int64_t w, int64_t h, int req_channels, unsigned char *out) {
    RasterCtx *ctx = (RasterCtx*)vctx;
    if (x < 0 || y < 0 || x + w > ctx->w || y + h > ctx->h) return -1;
    if (req_channels != ctx->c && !(req_channels == 1 && ctx->c >= 1)) return -2;

    for (int64_t row = 0; row < h; ++row) {
        const unsigned char *src = ctx->pixels + ((y + row) * ctx->w + x) * ctx->c;
        unsigned char *dst = out + row * (w * req_channels);
        if (req_channels == ctx->c) {
            memcpy(dst, src, (size_t)w * req_channels);
        } else if (req_channels == 1) {
            // simple luma from RGB
            for (int64_t col = 0; col < w; ++col) {
                const unsigned char *p = src + col*ctx->c;
                dst[col] = (unsigned char)((0.299*p[0] + 0.587*p[1] + 0.114*p[2]));
            }
        }
    }
    return 0;
}

static void raster_destroy(void *vctx) {
    RasterCtx *ctx = (RasterCtx*)vctx;
    if (ctx) {
        if (ctx->pixels) stbi_image_free(ctx->pixels);
        free(ctx);
    }
}

int raster_open(const char *path, ImageHandle *out) {
    int w=0,h=0,c=0;
    unsigned char *img = stbi_load(path, &w, &h, &c, 3);
    if (!img) return -1;
    RasterCtx *ctx = (RasterCtx*)calloc(1, sizeof(RasterCtx));
    ctx->pixels = img; ctx->w = w; ctx->h = h; ctx->c = 3;
    out->width = w; out->height = h; out->channels = 3;
    out->read_region = raster_read_region;
    out->destroy = raster_destroy;
    out->ctx = ctx;
    return 0;
}
