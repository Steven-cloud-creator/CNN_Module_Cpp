#ifndef __BLOCKCONV2D_H__
#define __BLOCKCONV2D_H__


#define MAX_WEIGHTS_CH_IN    8
#define MAX_WEIGHTS_CH_OUT   8
#define MAX_DATA_PAD_SIZE    77
#define MAX_DATA_SIZE        75
#define MAX_BIAS_CH_OUT      1024
// 定义数据类型和常量
typedef float D32;

// 卷积函数声明
void Conv(D32* data_in, D32 *weights, D32 *bias, D32 *data_out,
          int size, int ch_in, int ch_out, int stride, int padding, int groups);

#endif // __BLOCKCONV2D_H__
