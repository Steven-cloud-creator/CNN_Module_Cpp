#include <iostream>
#include <iomanip>
#include "../Modules/include/BlockConv2d.h"

int main() {
    // 测试参数
    int size = 150;           // 输入数据尺寸
    int ch_in = 16;         // 输入通道数
    int ch_out = 32;         // 输出通道数
    int stride = 2;         // 步长
    int groups = 16;         // 分组数
    int padding = 0;            // 填充大小
    
    // 计算输出尺寸
    int out_size = (size + 2 * padding - 3) / stride + 1;
    std::cout << "输入尺寸: " << size << "x" << size << std::endl;
    std::cout << "输出尺寸: " << out_size << "x" << out_size << std::endl;
    
    // 分配内存
    D32* data_in = (D32*)malloc(ch_in * size * size * sizeof(D32));
    D32* weights = (D32*)malloc(ch_out * ch_in * 9 * sizeof(D32));
    D32* bias = (D32*)malloc(ch_out * sizeof(D32));
    D32* data_out = (D32*)malloc(ch_out * out_size * out_size * sizeof(D32));
    
    if (!data_in || !weights || !bias || !data_out) {
        std::cout << "内存分配失败!" << std::endl;
        return -1;
    }
    
    // 初始化输入数据为1
    for (int c = 0; c < ch_in; c++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                data_in[c * size * size + i * size + j] = 1;
            }
        }
    }
    
    // 初始化权重为1
    for (int c_out = 0; c_out < ch_out; c_out++) {
        for (int c_in = 0; c_in < ch_in; c_in++) {
            for (int k = 0; k < 9; k++) {
                weights[c_out * ch_in * 9 + c_in * 9 + k] = 1;
            }
        }
    }
    
    // 初始化偏置为0
    for (int c = 0; c < ch_out; c++) {
        bias[c] = 0;
    }
    
    // 调用卷积函数
    std::cout << "开始卷积计算..." << std::endl;
    Conv(data_in,weights,bias,data_out,size,ch_in,ch_out,stride,padding,groups);
    std::cout << "卷积计算完成" << std::endl;
    
    // 打印部分输出结果
    std::cout << "\n部分输出结果:" << std::endl;
    int print_channels = (ch_out < 4) ? ch_out : 4;  // 打印前4个通道或所有通道
    int print_size = (out_size < 5) ? out_size : 5;  // 打印每个通道的前5x5个元素
    
    for (int c = 0; c < print_channels; c++) {
        std::cout << "通道 " << c << ":" << std::endl;
        for (int i = 0; i < print_size; i++) {
            for (int j = 0; j < print_size; j++) {
                std::cout << std::fixed << std::setprecision(2) 
                          << std::setw(8) << data_out[c * out_size * out_size + i * out_size + j];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 释放内存
    free(data_in);
    free(weights);
    free(bias);
    free(data_out);
    
    return 0;
}
