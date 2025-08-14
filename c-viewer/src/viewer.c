#include "viewer.h"
#include <SDL.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Forward decls
int raster_open(const char *path, ImageHandle *out);
int svs_open(const char *path, ImageHandle *out);
int dcm_open(const char *path, ImageHandle *out);

static int endswith(const char *s, const char *suf) {
    size_t ls = strlen(s), lu = strlen(suf);
    if (lu > ls) return 0;
    return _stricmp(s + (ls - lu), suf) == 0;
}

static int open_image(const char *path, ImageHandle *out) {
    if (endswith(path, ".png") || endswith(path, ".jpg") || endswith(path, ".jpeg")) {
        return raster_open(path, out);
    }
    if (endswith(path, ".svs")) {
        return svs_open(path, out);
    }
    if (endswith(path, ".dcm")) {
        return dcm_open(path, out);
    }
    return -1;
}

int viewer_run(const char *path) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL init failed: %s\n", SDL_GetError());
        return 1;
    }

    ImageHandle img = {0};
    if (open_image(path, &img) != 0) {
        fprintf(stderr, "Failed to open image: %s\n", path);
        SDL_Quit();
        return 2;
    }

    int winW = (int)(img.width > 1920 ? 1920 : img.width);
    int winH = (int)(img.height > 1080 ? 1080 : img.height);

    SDL_Window *win = SDL_CreateWindow("C Image Viewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, winW, winH, SDL_WINDOW_RESIZABLE);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    // Create streaming texture (RGB8)
    SDL_Texture *tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, (int)img.width, (int)img.height);

    int channel = -1; // -1 = RGB, 0=R,1=G,2=B

    int quit = 0;
    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = 1;
            if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_ESCAPE: quit = 1; break;
                    case SDLK_0: channel = -1; break;
                    case SDLK_1: channel = 0; break;
                    case SDLK_2: channel = 1; break;
                    case SDLK_3: channel = 2; break;
                }
            }
        }

        // Display full frame or a center crop if extremely large to avoid GPU texture limits
        int64_t viewW = img.width;
        int64_t viewH = img.height;
        const int64_t maxDim = 8192; // safe texture limit on many GPUs
        int64_t offX = 0, offY = 0;
        if (viewW > maxDim || viewH > maxDim) {
            viewW = viewW > maxDim ? maxDim : viewW;
            viewH = viewH > maxDim ? maxDim : viewH;
            offX = (img.width - viewW)/2;
            offY = (img.height - viewH)/2;
        }

        int reqC = channel == -1 ? 3 : 1;
        size_t bufSize = (size_t)viewW * (size_t)viewH * (size_t)reqC;
        unsigned char *buf = (unsigned char*)malloc(bufSize);
        if (!buf) break;
        if (img.read_region(img.ctx, offX, offY, viewW, viewH, reqC, buf) != 0) {
            free(buf); break;
        }

        if (channel >= 0 && reqC == 1) {
            // expand gray -> RGB for texture upload
            unsigned char *rgb = (unsigned char*)malloc((size_t)viewW*viewH*3);
            for (int64_t i=0;i<viewW*viewH;++i) {
                rgb[3*i+0]=buf[i]; rgb[3*i+1]=buf[i]; rgb[3*i+2]=buf[i];
            }
            SDL_UpdateTexture(tex, NULL, rgb, (int)(viewW*3));
            free(rgb);
        } else {
            SDL_UpdateTexture(tex, NULL, buf, (int)(viewW*3));
        }
        free(buf);

        SDL_RenderClear(ren);
        SDL_Rect dst;
        int w,h; SDL_GetWindowSize(win, &w, &h);
        dst.x = 0; dst.y = 0; dst.w = w; dst.h = h;
        SDL_RenderCopy(ren, tex, NULL, &dst);
        SDL_RenderPresent(ren);
    }

    if (tex) SDL_DestroyTexture(tex);
    if (ren) SDL_DestroyRenderer(ren);
    if (win) SDL_DestroyWindow(win);
    if (img.destroy) img.destroy(img.ctx);
    SDL_Quit();
    return 0;
}
