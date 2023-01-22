// Example parser for MNIST data
// Author: Didier Lime
// École Centrale de Nantes, 2018

#include <parser.h>

uint32_t make_uint32(byte buffer[])
{
    // make a 32-bit integer from the 4 input bytes
    return ((uint32_t) buffer[0] << 24) | ((uint32_t) buffer[1] << 16) | ((uint32_t) buffer[2] << 8) | (uint32_t) buffer[3];
}

byte* read_labels(const char filename[], unsigned* n )
{
    FILE* data = fopen(filename, "r");

    if (data == NULL)
    {
        perror(filename);
        exit(1);
    }

    byte buf[4];

    fread(buf, 1, 4, data); // magic number (discarded)
    fread(buf, 1, 4, data); // number of labels
    *n = make_uint32(buf);

    byte* ls = (byte*) calloc(*n, sizeof(byte));

    // Read n labels
    fread(ls, 1, *n, data);

    fclose(data);

    return ls;
}

image* read_images(const char filename[], unsigned* n )
{
    FILE* data = fopen(filename, "r");

    if (data == NULL)
    {
        perror(filename);
        exit(1);
    }

    byte buf[4];

    fread(buf, 1, 4, data); // magic number (discarded)
    fread(buf, 1, 4, data); // number of images
    *n = make_uint32(buf);

    fread(buf, 1, 4, data); // rows (discarded)
    fread(buf, 1, 4, data); // columns (discarded)

    image* is = (image*) calloc(*n, sizeof(image));

    // Read n images
    fread(is, 28*28, *n, data);

    fclose(data);

    return is;
}

// int main()
// {
//     unsigned n, ntest;
//     image* is = read_images("train-images-idx3-ubyte", &n);
//     printf("training images read\n");
//     byte* ls = read_labels("train-labels-idx1-ubyte", &n);
//     printf("training labels read\n");

//     image* is_test = read_images("t10k-images-idx3-ubyte", &ntest);
//     printf("test images read\n");
//     byte* ls_test = read_labels("t10k-labels-idx1-ubyte", &ntest);
//     printf("test labels read\n");

//     return 0;
// }
