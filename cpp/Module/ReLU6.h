#ifndef __RELU6_H__
#define __RELU6_H__

#include "Common.h"

class ReLU6 {
private:
    D32 threshold;  // ReLU6的阈值，默认6.0

public:
    // 构造函数：可指定阈值（默认6.0）
    ReLU6(D32 threshold = 6.0f);

    ~ReLU6();

    // 前向传播函数
    D32* forward(const D32* input, int total_size);
};

#endif