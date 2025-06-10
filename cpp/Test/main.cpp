#include <iostream>
#include "../Modules/include/ConvBNReLU.h"

void ConvBNReLU_Test();
void ExtendedChannel_Test();
int main()
{
    //ConvBNReLU_Test();
    ExtendedChannel_Test();
    return 0;
}

void ConvBNReLU_Test()
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

    //创建ConvBNReLU
    ConvBNReLU convbnrelu(in_channels,out_channels,kernel_size,stride,1,padding,true);
//==========初始化卷积层权重===============
    // 初始化卷积权重（全1）
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    D32 *weight = (D32 *)malloc(weight_size * sizeof(D32));
    for (int i = 0; i < weight_size; ++i)
    {
        weight[i] = 1.0f;
    }
    // 初始化偏置（全0）
    D32 *bias = (D32 *)malloc(out_channels * sizeof(D32));
    for (int i = 0; i < out_channels; ++i)
    {
        bias[i] = 0.0f;
    }
    // 设置卷积权重和偏置
    convbnrelu.m_pConv2d->set_weight(weight);
    convbnrelu.m_pConv2d->set_bias(bias);

    // 释放临时内存
    free(weight);
    free(bias);
//==========初始化BatchNorm统计量（模拟预训练值）===============
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
    convbnrelu.m_pBatchNorm2d->set_stats(bn_mean,bn_var,bn_weight,bn_bias);
    // 释放临时内存
    free(bn_mean);
    free(bn_var);
    free(bn_weight);
    free(bn_bias);

    // 模拟输入数据（全1）
    int input_size = batch_size * in_channels * in_height * in_width;
    D32 *input = (D32 *)malloc(input_size * sizeof(D32));
    for (int i = 0; i < input_size; ++i)
    {
        input[i] = 1.0f;
    }

    D32 *output = convbnrelu.forward(input,batch_size,in_height,in_width);
    // // 计算卷积输出尺寸
    int conv_out_height = convbnrelu.m_pConv2d->calculate_output_size(in_height);
    int conv_out_width = convbnrelu.m_pConv2d->calculate_output_size(in_width);
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
                    std::cout << output[idx] << "\t";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    free(input);
    free(output);
}

void ExtendedChannel_Test()
{
    // 配置参数
    int batch_size = 1;
    int in_channels = 1;
    int out_channels = 3;
    int kernel_size = 1;
    int in_height = 5;
    int in_width = 5;
    int stride = 1;
    int padding = 0;

    //创建Conv
    Conv2d conv(in_channels,out_channels,kernel_size,stride,padding,1,false);
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    D32 *weight = (D32 *)malloc(weight_size * sizeof(D32));
    for (int i = 0; i < weight_size; ++i)
    {
        weight[i] = 1.0f;
    }
    // 设置卷积权重和偏置
    conv.set_weight(weight);
    // 释放临时内存
    free(weight);
    // 模拟输入数据（全1）
    int input_size = batch_size * in_channels * in_height * in_width;
    D32 *input = (D32 *)malloc(input_size * sizeof(D32));
    for (int i = 0; i < input_size; ++i)
    {
        input[i] = 1.0f;
    }
    D32 *output = conv.forward(input,batch_size,in_height,in_width);
    // // 计算卷积输出尺寸
    int conv_out_height = conv.calculate_output_size(in_height);
    int conv_out_width = conv.calculate_output_size(in_width);
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
                    std::cout << output[idx] << "\t";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    free(input);
    free(output);
}