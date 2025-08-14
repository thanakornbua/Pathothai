#include "loader_common.h"
#include <stdint.h>
#include <stdlib.h>
#ifdef HAVE_OPENSLIDE
#include <openslide/openslide.h>
#endif

typedef struct {
#ifdef HAVE_OPENSLIDE
    openslide_t *osr;
#endif
    int64_t w, h;
} SVSCtx;

static int svs_read_region(void *vctx, int64_t x, int64_t y, int64_t w, int64_t h, int req_channels, unsigned char *out) {
#ifndef HAVE_OPENSLIDE
    (void)vctx; (void)x; (void)y; (void)w; (void)h; (void)req_channels; (void)out; return -1;
#else
    SVSCtx *ctx = (SVSCtx*)vctx;
    unsigned char *buf = (unsigned char*)malloc((size_t)w*h*4);
    if (!buf) return -2;
    openslide_read_region(ctx->osr, (uint32_t*)buf, x, y, 0, w, h);
    // convert ARGB -> RGB or Gray
    for (int64_t i=0;i<w*h;++i){
        unsigned char a = buf[4*i+0];
        unsigned char r = buf[4*i+1];
        unsigned char g = buf[4*i+2];
        unsigned char b = buf[4*i+3];
        if (req_channels == 3) {
            out[3*i+0] = r; out[3*i+1] = g; out[3*i+2] = b;
        } else if (req_channels == 1) {
            out[i] = (unsigned char)(0.299*r + 0.587*g + 0.114*b);
        }
    }
    free(buf);
    return 0;
#endif
}

static void svs_destroy(void *vctx) {
#ifdef HAVE_OPENSLIDE
    SVSCtx *ctx = (SVSCtx*)vctx;
    if (ctx && ctx->osr) openslide_close(ctx->osr);
#endif
    free(vctx);
}

int svs_open(const char *path, ImageHandle *out) {
#ifndef HAVE_OPENSLIDE
    (void)path; (void)out; return -1;
#else
    SVSCtx *ctx = (SVSCtx*)calloc(1, sizeof(SVSCtx));
    ctx->osr = openslide_open(path);
    if (!ctx->osr) { free(ctx); return -2; }
    int64_t w=0,h=0;
    w = openslide_get_level0_width(ctx->osr);
    h = openslide_get_level0_height(ctx->osr);
    ctx->w=w; ctx->h=h;
    out->width=w; out->height=h; out->channels=3;
    out->read_region = svs_read_region;
    out->destroy = svs_destroy;
    out->ctx = ctx;
    return 0;
#endif
}
