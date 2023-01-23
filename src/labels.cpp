#include <labels.hpp>

TP_MCIA::Labels::Labels(const char filename[], unsigned *n)
{
    m_n = *n;
    mp_labels = read_labels(filename, n);
}

TP_MCIA::Labels::~Labels()
{
    free(mp_labels);
}