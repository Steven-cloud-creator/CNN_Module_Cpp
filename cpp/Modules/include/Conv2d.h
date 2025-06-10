#ifndef __CONV2D_H__
#define __CONV2D_H__

#include "Common.h"

class Conv2d{

private:
    int in_channels;    //输入通道
    int out_channels;   //输出通道
    int kernel_size;    //卷积核大小
    int stride;         //步长
    int padding;        //填充大小
    int groups;         //分组
    bool bias_enabled;  //是否使用偏置

    float* weight;      //卷积核权重[out_channels,in_channels,kernel_size,kernel_size]
    float* bias;        //偏置 [out_channels]
public:
    //构造函数
    Conv2d(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int padding = 0, int groups = 1,bool bias = true);
    // 析构函数
    ~Conv2d();
    // 设置权重
    void set_weight(const float* new_weight);
    // 设置偏置
    void set_bias(const float* new_bias);
    // 获取输出通道
    int get_out_channels(void);
    // 计算输出尺寸
    int calculate_output_size(int input_size);
    // 前向传播
    float* forward(const float* input, int batch_size, int in_height, int in_width);
};

#endif