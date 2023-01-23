#ifndef LABELS_H
#define LABELS_H

#include <parser.h>

namespace TP_MCIA
{
    class Labels
    {
    public:
        Labels(const char filename[], unsigned *n);
        ~Labels();
        byte *mp_labels;
        unsigned m_n;
    };
};

#endif