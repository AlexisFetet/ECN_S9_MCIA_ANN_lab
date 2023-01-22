#ifndef PARSER_H
#define PARSER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef uint8_t byte;
typedef byte image[28*28];

uint32_t make_uint32(byte buffer[]);
byte *read_labels(const char filename[], unsigned *n);
image *read_images(const char filename[], unsigned *n);

#ifdef __cplusplus
}
#endif

#endif