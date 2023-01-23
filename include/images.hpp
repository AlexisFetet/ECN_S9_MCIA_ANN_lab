#ifndef IMAGES_H
#define IMAGES_H

#include <parser.h>

namespace TP_MCIA
{
    class Images
    {
    public:
        Images(const char filename[], unsigned *n);
        ~Images();
        image *mp_images;
        unsigned m_n;
    };
};

#endif