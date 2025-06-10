#ifndef __CONVBNRELU_H__
#define __CONVBNRELU_H__

#include "Common.h"
#include "Conv2d.h"
#include "BatchNorm2d.h"
#include "ReLU6.h"

class ConvBNReLU{
public:
    Conv2d* m_pConv2d;
    BatchNorm2d* m_pBatchNorm2d;
    ReLU6* m_pReLU6;

    ConvBNReLU(int in_channels, int out_channels,int kernel_size,
                int stride,int groups,int padding = 0,bool bias = false);
    ~ConvBNReLU();
    D32* forward(const D32* input,int batch_size,int in_height,int in_width);
};

#endif
