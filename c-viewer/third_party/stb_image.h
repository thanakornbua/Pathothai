// stb_image - public domain header for loading PNG/JPG
// Minimal vendored copy notice: https://github.com/nothings/stb
// Define STB_IMAGE_IMPLEMENTATION in one C file before including
#ifndef STB_IMAGE_H_INCLUDED
#define STB_IMAGE_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

extern const char *stbi_failure_reason(void);

unsigned char *stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp);
void stbi_image_free(void *retval_from_stbi_load);

#ifdef __cplusplus
}
#endif

#endif // STB_IMAGE_H_INCLUDED
