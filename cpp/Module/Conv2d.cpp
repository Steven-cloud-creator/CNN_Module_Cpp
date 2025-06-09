#include "Conv2d.h"

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size,
               int stride, int padding, bool bias)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding),
      bias_enabled(bias)
{
    // 计算权重参数数量
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    this->weight = (float *)malloc(weight_size * sizeof(float));

    if (bias_enabled)
    {
        this->bias = (float *)malloc(out_channels * sizeof(float));
    }
    else
    {
        this->bias = nullptr;
    }

    // 初始化为零（实际使用中应使用合适的初始化方法）
    std::memset(this->weight, 0, weight_size * sizeof(float));
    if (bias_enabled)
    {
        std::memset(this->bias, 0, out_channels * sizeof(float));
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
void Conv2d::set_weight(const float *new_weight)
{
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    std::memcpy(weight, new_weight, weight_size * sizeof(float));
}
// 设置偏置
void Conv2d::set_bias(const float *new_bias)
{
    if (bias_enabled)
    {
        std::memcpy(bias, new_bias, out_channels * sizeof(float));
    }
}
// 计算输出尺寸
int Conv2d::calculate_output_size(int input_size)
{
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}
// 前向传播
float *Conv2d::forward(const float *input, int batch_size, int in_height, int in_width)
{
    // 计算输出尺寸
    int out_height = calculate_output_size(in_height);
    int out_width = calculate_output_size(in_width);

    // 分配输出内存
    int output_size = batch_size * out_channels * out_height * out_width;
    float *output = (float *)malloc(output_size * sizeof(float));
    std::memset(output, 0, output_size * sizeof(float));

    // 为输入添加填充（如果有）
    float *padded_input = nullptr;
    int padded_height = in_height + 2 * padding;
    int padded_width = in_width + 2 * padding;

    if (padding > 0)
    {
        int padded_size = batch_size * in_channels * padded_height * padded_width;
        padded_input = (float *)malloc(padded_size * sizeof(float));
        std::memset(padded_input, 0, padded_size * sizeof(float));

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
        padded_input = (float *)input; // 无填充时直接使用原始输入
    }

    // 执行卷积
    for (int b = 0; b < batch_size; ++b)
    { // 遍历批次
        for (int oc = 0; oc < out_channels; ++oc)
        { // 遍历输出通道
            for (int oh = 0; oh < out_height; ++oh)
            { // 遍历输出高度
                for (int ow = 0; ow < out_width; ++ow)
                { // 遍历输出宽度
                    // 计算当前输出位置对应的输入区域
                    int h_start = oh * stride;
                    int w_start = ow * stride;

                    float sum = 0.0f;

                    // 遍历所有输入通道
                    for (int ic = 0; ic < in_channels; ++ic)
                    {
                        // 遍历卷积核
                        for (int kh = 0; kh < kernel_size; ++kh)
                        {
                            for (int kw = 0; kw < kernel_size; ++kw)
                            {
                                // 计算输入位置
                                int h = h_start + kh;
                                int w = w_start + kw;

                                // 计算权重索引
                                int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                                 ic * kernel_size * kernel_size +
                                                 kh * kernel_size +
                                                 kw;

                                // 计算输入索引（考虑填充）
                                int input_idx = b * in_channels * padded_height * padded_width +
                                                ic * padded_height * padded_width +
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
                        sum += bias[oc];
                    }

                    // 存储结果
                    int output_idx = b * out_channels * out_height * out_width +
                                     oc * out_height * out_width +
                                     oh * out_width +
                                     ow;

                    output[output_idx] = sum;
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

// #define INPUT_CHANNEL 1
// #define INPUT_HEIGHT 5
// #define INPUT_WIDTH 5
// #define KERNEL_SIZE 3
// #define OUTPUT_CHANNEL 3
// int main()
// {
//     // 分配内存
//     D32* in = (D32*)malloc(INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH * sizeof(D32));
//     D32* weight = (D32*)malloc(OUTPUT_CHANNEL * INPUT_CHANNEL * KERNEL_SIZE * KERNEL_SIZE * sizeof(D32));
//     D32* bias = (D32*)malloc(OUTPUT_CHANNEL * sizeof(D32));
//     // 计算输出尺寸
//     int padding = 1;
//     int stride = 2;
//     int out_height = (INPUT_HEIGHT + 2 * padding - KERNEL_SIZE) / stride + 1;
//     int out_width = (INPUT_WIDTH + 2 * padding - KERNEL_SIZE) / stride + 1;
//     D32* out = (D32*)malloc(OUTPUT_CHANNEL * out_height * out_width * sizeof(D32));
//     // 检查内存分配是否成功
//     if (in == nullptr || weight == nullptr || bias == nullptr) {
//         return 1;
//     }
//     // 初始化输入数据 - 全部设为1
//     for(int k = 0; k < INPUT_CHANNEL; k++) {
//         for(int i = 0; i < INPUT_HEIGHT; i++) {
//             for(int j = 0; j < INPUT_WIDTH; j++) {
//                 in[k * INPUT_HEIGHT * INPUT_WIDTH + i * INPUT_WIDTH + j] = 1;
//             }
//         }
//     }

//     // 初始化权重数据 - 全部设为1
//     for(int l = 0; l < OUTPUT_CHANNEL; l++) {
//         for(int k = 0; k < INPUT_CHANNEL; k++) {
//             for(int i = 0; i < KERNEL_SIZE; i++) {
//                 for(int j = 0; j < KERNEL_SIZE; j++) {
//                     weight[l * INPUT_CHANNEL * KERNEL_SIZE * KERNEL_SIZE + k * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j] = 1;
//                 }
//             }
//         }
//     }

//     //二维卷积
//     Conv2d(in,INPUT_CHANNEL,INPUT_HEIGHT,INPUT_WIDTH,out,OUTPUT_CHANNEL,weight,bias,KERNEL_SIZE,stride,padding,1);

//     // 输出卷积结果 (只输出第一个通道的结果作为示例)
//     cout << "Output shape: " << OUTPUT_CHANNEL << "x" << out_height << "x" << out_width << endl;
//     cout << "First channel output:" << endl;

//     for(int i = 0; i < min(10, out_height); i++) {  // 只输出前10行
//         for(int j = 0; j < min(10, out_width); j++) {  // 只输出前10列
//             cout << (float)out[(i) * out_width + j] << "\t";
//         }
//         cout << endl;
//     }
//     // 释放内存
//     free(in);
//     free(weight);
//     free(bias);
//     free(out);

//     return 0;
// }