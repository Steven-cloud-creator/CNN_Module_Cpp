#include "BatchNorm2d.h"

BatchNorm2d::BatchNorm2d(int num_features, 
                const D32* mean,
                const D32* var,
                const D32* weight,
                const D32* bias,
                D32 eps)
    :num_features(num_features), eps(eps)
{
// 分配内存
    running_mean = (D32*)malloc(num_features * sizeof(D32));
    running_var = (D32*)malloc(num_features * sizeof(D32));
    
    // 初始化默认值（避免空指针风险）
    for (int i = 0; i < num_features; ++i) {
        running_mean[i] = 0.0f;
        running_var[i] = 1.0f;
    }
    
    // 如果提供了 mean 和 var，则复制
    if(mean != nullptr && var != nullptr) {
        for (int i = 0; i < num_features; ++i) {
            running_mean[i] = mean[i];
            running_var[i] = var[i];
        }
    }
    
    // 处理可学习参数
    affine = (weight != nullptr && bias != nullptr);
    if (affine) {
        this->weight = (D32*)malloc(num_features * sizeof(D32));
        this->bias = (D32*)malloc(num_features * sizeof(D32));
        for (int i = 0; i < num_features; ++i) {
            this->weight[i] = weight[i];
            this->bias[i] = bias[i];
        }
    } else {
        this->weight = nullptr;
        this->bias = nullptr;
    }
}

BatchNorm2d::~BatchNorm2d()
{
    free(running_mean);
    free(running_var);
    if(affine){
        free(weight);
        free(bias);
    }
}

// 设置统计量的方法
void BatchNorm2d::set_stats(const D32* mean, const D32* var, 
                           const D32* weight, const D32* bias)
{
    if (mean != nullptr && var != nullptr) {
        for (int i = 0; i < num_features; ++i) {
            running_mean[i] = mean[i];
            running_var[i] = var[i];
        }
    }
    
    if (affine && weight != nullptr && bias != nullptr) {
        for (int i = 0; i < num_features; ++i) {
            this->weight[i] = weight[i];
            this->bias[i] = bias[i];
        }
    }
}

D32* BatchNorm2d::forward(const D32* input, int batch_size, int height, int width)
{
 int total_size = batch_size * num_features * height * width;
    D32* output = (D32*)malloc(total_size * sizeof(D32));

    int channel_size = height * width;
        
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_features; ++c) {
            int channel_offset = b * num_features * channel_size + c * channel_size;
                
            D32 mean = running_mean[c];
            D32 var = running_var[c];
            D32 scale = affine ? weight[c] : 1.0f;
            D32 shift = affine ? bias[c] : 0.0f;
                
            for (int i = 0; i < channel_size; ++i) {
                int idx = channel_offset + i;
                output[idx] = scale * (input[idx] - mean) / std::sqrt(var + eps) + shift;
            }
        }
    }
    
    return output;
}