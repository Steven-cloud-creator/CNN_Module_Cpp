#include <iostream>
#include "Module/Conv2d.h"
#include "Module/BatchNorm2d.h"  // 添加BatchNorm2d头文件
#include "Module/ReLU6.h"  // 添加ReLU6头文件

int main()
{
    // 配置参数
    int batch_size = 1;
    int in_channels = 1;
    int out_channels = 3;
    int kernel_size = 3;
    int in_height = 5;
    int in_width = 5;
    int stride = 2;
    int padding = 1;

    // 创建Conv2d层
    Conv2d conv(in_channels, out_channels, kernel_size, stride, padding, true);

    // 初始化卷积权重（全1）
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    float *weight = (float *)malloc(weight_size * sizeof(float));
    for (int i = 0; i < weight_size; ++i)
    {
        weight[i] = 1.0f;
    }

    // 初始化偏置（全0）
    float *bias = (float *)malloc(out_channels * sizeof(float));
    for (int i = 0; i < out_channels; ++i)
    {
        bias[i] = 0.0f;
    }

    // 设置卷积权重和偏置
    conv.set_weight(weight);
    conv.set_bias(bias);

    // 释放临时内存
    free(weight);
    free(bias);

    // 模拟输入数据（全1）
    int input_size = batch_size * in_channels * in_height * in_width;
    float *input = (float *)malloc(input_size * sizeof(float));
    for (int i = 0; i < input_size; ++i)
    {
        input[i] = 1.0f;
    }

    // 执行卷积
    float *conv_output = conv.forward(input, batch_size, in_height, in_width);

    // 计算卷积输出尺寸
    int conv_out_height = conv.calculate_output_size(in_height);
    int conv_out_width = conv.calculate_output_size(in_width);

    // 打印卷积输出形状
    std::cout << "Conv Output shape: ["
              << batch_size << ", "
              << out_channels << ", "
              << conv_out_height << ", "
              << conv_out_width << "]" << std::endl;

    // 初始化BatchNorm统计量（模拟预训练值）
    D32 *bn_mean = (D32 *)malloc(out_channels * sizeof(D32));
    D32 *bn_var = (D32 *)malloc(out_channels * sizeof(D32));
    D32 *bn_weight = (D32 *)malloc(out_channels * sizeof(D32));
    D32 *bn_bias = (D32 *)malloc(out_channels * sizeof(D32));
    
    for (int i = 0; i < out_channels; ++i) {
        bn_mean[i] = 5.4444;     // 假设均值为0
        bn_var[i] = 1.3334;      // 假设方差为1
        bn_weight[i] = 1.0f;   // 缩放因子
        bn_bias[i] = 0.0f;     // 偏移量
    }
    // 创建BatchNorm2d层，使用卷积的输出通道数
    BatchNorm2d bn(out_channels, 
                  bn_mean,  // 运行时均值（将在set_stats中设置）
                  bn_var,  // 运行时方差（将在set_stats中设置）
                  bn_weight,  // 权重（将在set_stats中设置）
                  bn_bias,  // 偏置（将在set_stats中设置）
                  1e-5);

    // 释放临时内存
    free(bn_mean);
    free(bn_var);
    free(bn_weight);
    free(bn_bias);

    // 执行BatchNorm
    float *bn_output = bn.forward(conv_output, batch_size, conv_out_height, conv_out_width);

    // 创建ReLU6层（使用默认阈值6.0）
    ReLU6 relu6;
    // 执行ReLU6
    float *relu_output = relu6.forward(bn_output, batch_size * out_channels * conv_out_height * conv_out_width);


    // 打印ReLU输出内容
    std::cout << "Final Output (after ReLU6) content:" << std::endl;
    for (int b = 0; b < batch_size; ++b)
    {
        std::cout << "Batch " << b << ":\n";
        for (int oc = 0; oc < out_channels; ++oc)
        {
            std::cout << "  Channel " << oc << ":\n";
            for (int h = 0; h < conv_out_height; ++h)
            {
                std::cout << "    ";
                for (int w = 0; w < conv_out_width; ++w)
                {
                    int idx = b * out_channels * conv_out_height * conv_out_width +
                            oc * conv_out_height * conv_out_width +
                            h * conv_out_width +
                            w;
                    std::cout << relu_output[idx] << "\t";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    // 释放内存
    free(input);
    free(conv_output);  // 释放卷积输出
    free(bn_output);    // 释放BatchNorm输出
    free(relu_output);  // 释放ReLU输出

    return 0;
}