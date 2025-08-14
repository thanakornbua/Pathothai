#include "viewer.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: c_image_viewer <image.{png|jpg|jpeg|svs|dcm}>\n");
        return 0;
    }
    return viewer_run(argv[1]);
}
