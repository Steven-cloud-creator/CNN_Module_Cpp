#include "ReLU6.h"

ReLU6::ReLU6(D32 threshold) : threshold(threshold)
{

}
ReLU6::~ReLU6() {
    // 无动态分配内存，析构函数为空
}
D32* ReLU6::forward(const D32* input, int total_size) {
    D32* output = (D32*)malloc(total_size * sizeof(D32));
    for (int i = 0; i < total_size; ++i) {
        // 应用ReLU6公式：max(0, min(x, threshold))
        output[i] = (input[i] < 0.0f) ? 0.0f : ((input[i] > threshold) ? threshold : input[i]);
    }
    return output;
}
