#ifndef __BATCHNORM2D_H__
#define __BATCHNORM2D_H__

#include "Common.h"

class BatchNorm2d{

private:
    int num_features;
    D32 eps;
    bool affine;

    D32* weight;        //缩放因子
    D32* bias;          //偏移因子
    D32* running_mean;  //全局均值
    D32* running_var;    //全局方差
public:
    //构造函数
    BatchNorm2d(int num_features, 
                const D32* mean,
                const D32* var,
                const D32* weight = nullptr,
                const D32* bias = nullptr,
                D32 eps = 1e-5);

    ~BatchNorm2d();

    void set_stats(const D32* mean, const D32* var, 
                  const D32* weight = nullptr, const D32* bias = nullptr);

    D32* forward(const D32* input, int batch_size, int height, int width);
        
};

#endif