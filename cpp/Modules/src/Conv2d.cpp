#include "../include/Conv2d.h"

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size,
               int stride, int padding, int groups, bool bias)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding),
      groups(groups), bias_enabled(bias)
{
    // 验证groups参数的有效性
    if (in_channels % groups != 0 || out_channels % groups != 0) {
        throw std::invalid_argument("in_channels and out_channels must be divisible by groups");
    }

    // 计算每组的输入/输出通道数
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    // 计算权重参数数量
    int weight_size = out_channels * in_channels_per_group * kernel_size * kernel_size;
    this->weight = (D32 *)malloc(weight_size * sizeof(D32));

    if (bias_enabled)
    {
        this->bias = (D32 *)malloc(out_channels * sizeof(D32));
    }
    else
    {
        this->bias = nullptr;
    }

    // 初始化为零（实际使用中应使用合适的初始化方法）
    std::memset(this->weight, 0, weight_size * sizeof(D32));
    if (bias_enabled)
    {
        std::memset(this->bias, 0, out_channels * sizeof(D32));
    }
}

Conv2d::~Conv2d()
{
    free(weight);
    if (bias_enabled)
    {
        free(bias);
    }
}

void Conv2d::set_weight(const D32 *new_weight)
{
    int weight_size = out_channels * (in_channels / groups) * kernel_size * kernel_size;
    std::memcpy(weight, new_weight, weight_size * sizeof(D32));
}

// 设置偏置
void Conv2d::set_bias(const D32 *new_bias)
{
    if (bias_enabled)
    {
        std::memcpy(bias, new_bias, out_channels * sizeof(D32));
    }
}

int Conv2d::get_out_channels(void)
{
    return out_channels;
}

// 计算输出尺寸
int Conv2d::calculate_output_size(int input_size)
{
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

// 前向传播
float *Conv2d::forward(const D32 *input, int batch_size, int in_height, int in_width)
{
    // 计算输出尺寸
    int out_height = calculate_output_size(in_height);
    int out_width = calculate_output_size(in_width);

    // 分配输出内存
    int output_size = batch_size * out_channels * out_height * out_width;
    D32 *output = (D32 *)malloc(output_size * sizeof(D32));
    std::memset(output, 0, output_size * sizeof(D32));

    // 为输入添加填充（如果有）
    D32 *padded_input = nullptr;
    int padded_height = in_height + 2 * padding;
    int padded_width = in_width + 2 * padding;

    if (padding > 0)
    {
        int padded_size = batch_size * in_channels * padded_height * padded_width;
        padded_input = (D32 *)malloc(padded_size * sizeof(D32));
        std::memset(padded_input, 0, padded_size * sizeof(D32));

        // 复制原始输入到填充后的输入（中心区域）
        for (int b = 0; b < batch_size; ++b)
        {
            for (int c = 0; c < in_channels; ++c)
            {
                for (int h = 0; h < in_height; ++h)
                {
                    for (int w = 0; w < in_width; ++w)
                    {
                        int padded_idx = b * in_channels * padded_height * padded_width +
                                         c * padded_height * padded_width +
                                         (h + padding) * padded_width +
                                         (w + padding);

                        int input_idx = b * in_channels * in_height * in_width +
                                        c * in_height * in_width +
                                        h * in_width +
                                        w;

                        padded_input[padded_idx] = input[input_idx];
                    }
                }
            }
        }
    }
    else
    {
        padded_input = (D32 *)input; // 无填充时直接使用原始输入
    }

    // 每组的输入/输出通道数
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    // 执行分组卷积
    for (int b = 0; b < batch_size; ++b)
    { // 遍历批次
        for (int g = 0; g < groups; ++g)
        { // 遍历分组
            for (int oc = 0; oc < out_channels_per_group; ++oc)
            { // 遍历当前组的输出通道
                int global_oc = g * out_channels_per_group + oc; // 全局输出通道索引
                
                for (int oh = 0; oh < out_height; ++oh)
                { // 遍历输出高度
                    for (int ow = 0; ow < out_width; ++ow)
                    { // 遍历输出宽度
                        // 计算当前输出位置对应的输入区域
                        int h_start = oh * stride;
                        int w_start = ow * stride;

                        float sum = 0.0f;

                        // 遍历当前组的输入通道
                        for (int ic = 0; ic < in_channels_per_group; ++ic)
                        {
                            int global_ic = g * in_channels_per_group + ic; // 全局输入通道索引
                            
                            // 遍历卷积核
                            for (int kh = 0; kh < kernel_size; ++kh)
                            {
                                for (int kw = 0; kw < kernel_size; ++kw)
                                {
                                    // 计算输入位置
                                    int h = h_start + kh;
                                    int w = w_start + kw;

                                    // 计算权重索引
                                    int weight_idx = global_oc * in_channels_per_group * kernel_size * kernel_size +
                                                     ic * kernel_size * kernel_size +
                                                     kh * kernel_size +
                                                     kw;

                                    // 计算输入索引（考虑填充）
                                    int input_idx = b * in_channels * padded_height * padded_width +
                                                    global_ic * padded_height * padded_width +
                                                    h * padded_width +
                                                    w;

                                    // 累加卷积结果
                                    sum += padded_input[input_idx] * weight[weight_idx];
                                }
                            }
                        }

                        // 添加偏置（如果有）
                        if (bias_enabled)
                        {
                            sum += bias[global_oc];
                        }

                        // 存储结果
                        int output_idx = b * out_channels * out_height * out_width +
                                         global_oc * out_height * out_width +
                                         oh * out_width +
                                         ow;

                        output[output_idx] = sum;
                    }
                }
            }
        }
    }

    // 释放填充的输入（如果有）
    if (padding > 0)
    {
        free(padded_input);
    }

    return output;
}