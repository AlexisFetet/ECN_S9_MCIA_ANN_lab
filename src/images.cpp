#include <images.hpp>

TP_MCIA::Images::Images(const char filename[], unsigned *n)
{
    m_n = *n;
    mp_images = read_images(filename, n);
}

TP_MCIA::Images::~Images()
{
    free(mp_images);
}