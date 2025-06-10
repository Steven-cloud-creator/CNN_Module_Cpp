#include "../include/ConvBNReLU.h"

ConvBNReLU::ConvBNReLU(int in_channels,int out_channels,int kernel_size,
                        int stride,int groups,int padding,bool bias)
{
    m_pConv2d = new Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias);
    D32* mean = nullptr;
    D32* var = nullptr;
    D32* bn_weight = nullptr;
    D32* bn_bias = nullptr;
    D32 eps = 1e-5;
    m_pBatchNorm2d = new BatchNorm2d(out_channels, mean, var, bn_weight, bn_bias, eps);
    m_pReLU6 = new ReLU6();
}
ConvBNReLU::~ConvBNReLU()
{
    delete m_pConv2d;
    delete m_pBatchNorm2d;
    delete m_pReLU6;
}
D32* ConvBNReLU::forward(const D32* input, int batch_size, int in_height, int in_width)
{
    // 卷积层前向传播
    D32* conv_output = m_pConv2d->forward(input, batch_size, in_height, in_width);

    // 计算卷积层输出的高度和宽度
    int out_height = m_pConv2d->calculate_output_size(in_height);
    int out_width = m_pConv2d->calculate_output_size(in_width);

    // 批量归一化层前向传播
    D32* bn_output = m_pBatchNorm2d->forward(conv_output, batch_size, out_height, out_width);

    // 释放卷积层输出的内存
    free(conv_output);

    // 计算批量归一化层输出的总大小
    int total_size = batch_size * m_pConv2d->get_out_channels() * out_height * out_width;

    // ReLU6 层前向传播
    D32* relu_output = m_pReLU6->forward(bn_output, total_size);

    // 释放批量归一化层输出的内存
    free(bn_output);

    return relu_output;
}
