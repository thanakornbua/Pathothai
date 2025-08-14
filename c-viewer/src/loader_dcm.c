#include "loader_common.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_GDCM
#include <gdcmImageReader.h>
#include <gdcmImage.h>
#include <gdcmSmartPointer.h>
#endif

typedef struct {
    unsigned char *pixels; // RGB8
    int64_t w,h;
    int c;
} DcmCtx;

static int dcm_read_region(void *vctx, int64_t x, int64_t y, int64_t w, int64_t h, int req_channels, unsigned char *out) {
    DcmCtx *ctx = (DcmCtx*)vctx;
    if (x < 0 || y < 0 || x + w > ctx->w || y + h > ctx->h) return -1;
    // simple memcopy assuming row-major RGB8 or Gray8
    for (int64_t row=0; row<h; ++row) {
        const unsigned char *src = ctx->pixels + ((y+row)*ctx->w + x) * ctx->c;
        unsigned char *dst = out + row * (w * (req_channels==1?1:ctx->c));
        if (req_channels == 1 && ctx->c == 3) {
            for (int64_t col=0; col<w; ++col) {
                const unsigned char *p = src + col*3;
                dst[col] = (unsigned char)(0.299*p[0] + 0.587*p[1] + 0.114*p[2]);
            }
        } else {
            memcpy(dst, src, (size_t)w*ctx->c);
        }
    }
    return 0;
}

static void dcm_destroy(void *vctx) {
    DcmCtx *ctx = (DcmCtx*)vctx;
    if (ctx) {
        free(ctx->pixels);
        free(ctx);
    }
}

int dcm_open(const char *path, ImageHandle *out) {
#ifndef HAVE_GDCM
    (void)path; (void)out; return -1;
#else
    gdcm::ImageReader reader;
    reader.SetFileName(path);
    if (!reader.Read()) return -2;
    const gdcm::Image &image = reader.GetImage();
    unsigned int dims[3] = {0,0,0};
    image.GetDimensions(dims);
    size_t w = dims[0];
    size_t h = dims[1];
    size_t samples = image.GetNumberOfSamplesPerPixel();
    size_t bpp = image.GetPixelFormat().GetBitsAllocated();
    if (bpp != 8) return -3; // keep simple
    size_t c = samples >= 3 ? 3 : 1;
    size_t bytes = w*h*samples;
    unsigned char *buf = (unsigned char*)malloc(bytes);
    if (!buf) return -4;
    image.GetBuffer((char*)buf);
    DcmCtx *ctx = (DcmCtx*)calloc(1,sizeof(DcmCtx));
    ctx->pixels = buf; ctx->w=(int64_t)w; ctx->h=(int64_t)h; ctx->c=(int)c;
    out->width=ctx->w; out->height=ctx->h; out->channels=(int)c;
    out->read_region = dcm_read_region;
    out->destroy = dcm_destroy;
    out->ctx = ctx;
    return 0;
#endif
}
