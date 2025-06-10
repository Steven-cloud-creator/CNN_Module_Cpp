#include "../include/InvertedResidual.h"

InvertedResidual::InvertedResidual(int inp, int oup, int stride, float expand_ratio)
    : m_inp(inp), m_oup(oup), m_stride(stride) {
    assert(stride == 1 || stride == 2);

    m_hidden_dim = static_cast<int>(std::round(inp * expand_ratio));
    m_use_res_connect = (stride == 1 && inp == oup);

    // 初始化 expand 层（如果需要）
    if (expand_ratio != 1) {
        m_pExpandConv = new ConvBNReLU(inp, m_hidden_dim, 1, 1, 1);
    } else {
        m_pExpandConv = nullptr;
    }

    // 初始化 depthwise 层
    m_pDepthwiseConv = new ConvBNReLU(m_hidden_dim, m_hidden_dim, 3, stride, m_hidden_dim);

    // 初始化 project 层
    m_pProjectConv = new Conv2d(m_hidden_dim, oup, 1, 1, 0, 1, false);
    
    // 初始化 project BN 层
    D32* bn_weight = new D32[oup];
    D32* bn_bias = new D32[oup];
    D32* mean = new D32[oup];
    D32* var = new D32[oup];
    
    for (int i = 0; i < oup; ++i) {
        bn_weight[i] = 1.0f;
        bn_bias[i] = 0.0f;
        mean[i] = 0.0f;
        var[i] = 1.0f;
    }
    
    m_pProjectBN = new BatchNorm2d(oup, mean, var, bn_weight, bn_bias);
    
    delete[] bn_weight;
    delete[] bn_bias;
    delete[] mean;
    delete[] var;
}

InvertedResidual::~InvertedResidual() {
    if (m_pExpandConv) delete m_pExpandConv;
    delete m_pDepthwiseConv;
    delete m_pProjectConv;
    delete m_pProjectBN;
}

D32* InvertedResidual::forward(const D32* input, int batch_size, int in_height, int in_width) {
    D32* x = const_cast<D32*>(input);
    int current_height = in_height;
    int current_width = in_width;

    // Expand 层前向传播
    if (m_pExpandConv) {
        x = m_pExpandConv->forward(x, batch_size, current_height, current_width);
        current_height = m_pExpandConv->m_pConv2d->calculate_output_size(current_height);
        current_width = m_pExpandConv->m_pConv2d->calculate_output_size(current_width);
    }

    // Depthwise 层前向传播
    x = m_pDepthwiseConv->forward(x, batch_size, current_height, current_width);
    current_height = m_pDepthwiseConv->m_pConv2d->calculate_output_size(current_height);
    current_width = m_pDepthwiseConv->m_pConv2d->calculate_output_size(current_width);

    // Project 层前向传播
    float* project_output = m_pProjectConv->forward(x, batch_size, current_height, current_width);
    D32* bn_output = m_pProjectBN->forward(project_output, batch_size, current_height, current_width);
    
    // 释放中间结果
    free(project_output);

    // 处理残差连接
    if (m_use_res_connect) {
        // 创建用于存储结果的内存
        int output_size = batch_size * m_oup * current_height * current_width;
        D32* result = new D32[output_size];
        
        // 执行残差连接：x + conv(x)
        for (int i = 0; i < output_size; ++i) {
            result[i] = input[i] + bn_output[i];
        }
        
        // 释放输入的 bn_output
        free(bn_output);
        return result;
    } else {
        return bn_output;
    }
}