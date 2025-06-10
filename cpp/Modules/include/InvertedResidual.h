#ifndef __INVERTEDRESIDUAL_H__
#define __INVERTEDRESIDUAL_H__

#include "Common.h"
#include "ConvBNReLU.h"

class InvertedResidual{
private:
    ConvBNReLU* m_pExpandConv;
    ConvBNReLU* m_pDepthwiseConv;
    Conv2d* m_pProjectConv;
    BatchNorm2d* m_pProjectBN;
    int m_inp;
    int m_oup;
    int m_stride;
    bool m_use_res_connect;
    int m_hidden_dim;
public:
    InvertedResidual(int inp, int oup, int stride, float expand_ratio);
    ~InvertedResidual();
    D32* forward(const D32* input, int batch_size, int in_height, int in_width);
};

#endif