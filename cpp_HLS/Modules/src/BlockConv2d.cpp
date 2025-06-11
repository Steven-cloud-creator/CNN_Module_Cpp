#include "../include/BlockConv2d.h"

static D32 weights_buffer[MAX_WEIGHTS_CH_OUT][MAX_WEIGHTS_CH_IN][3][3];
static D32 bias_buffer[MAX_BIAS_CH_OUT];
static D32 data_in_buffer[MAX_WEIGHTS_CH_IN][MAX_DATA_PAD_SIZE][MAX_DATA_PAD_SIZE];
static D32 data_out_buffer[MAX_WEIGHTS_CH_OUT][MAX_DATA_SIZE][MAX_DATA_SIZE];

void Conv(D32* data_in, D32 *weights, D32 *bias, D32 *data_out,
          int size, int ch_in, int ch_out, int stride, int padding, int groups)
{
#pragma HLS ARRAY_PARTITION variable=data_out_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weights_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weights_buffer dim=2 complete
#pragma HLS ARRAY_PARTITION variable=data_in_buffer dim=1 complete
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=size
#pragma HLS INTERFACE s_axilite port=ch_in
#pragma HLS INTERFACE s_axilite port=ch_out
#pragma HLS INTERFACE s_axilite port=stride
#pragma HLS INTERFACE s_axilite port=padding
#pragma HLS INTERFACE s_axilite port=groups
#pragma HLS INTERFACE m_axi port=data_in bundle=DATA_IN
#pragma HLS INTERFACE m_axi port=weights bundle=WEIGHTS
#pragma HLS INTERFACE m_axi port=bias bundle=BIAS
#pragma HLS INTERFACE m_axi port=data_out bundle=DATA_OUT

    // 计算输出尺寸
    int out_size = (size - 3 + 2 * padding) / stride + 1;
    if (out_size <= 0) out_size = 1;  // 确保最小输出尺寸为1

    // 下载偏置到PL端
    for (int k = 0; k < ch_out; k++) {
        bias_buffer[k] = bias[k];
    }

    // 计算分组参数
    int ch_in_group = ch_in / groups;
    int ch_out_group = ch_out / groups;

    // 计算分块数量（考虑边界情况）
    int num_blocks_h = (out_size + MAX_DATA_SIZE - 1) / MAX_DATA_SIZE;
    int num_blocks_w = (out_size + MAX_DATA_SIZE - 1) / MAX_DATA_SIZE;
    int ch_in_sets = (ch_in_group + MAX_WEIGHTS_CH_IN - 1) / MAX_WEIGHTS_CH_IN;
    int ch_out_sets = (ch_out_group + MAX_WEIGHTS_CH_OUT - 1) / MAX_WEIGHTS_CH_OUT;

    // 初始化输出缓冲区
    for (int ko = 0; ko < MAX_WEIGHTS_CH_OUT; ko++) {
        for (int i = 0; i < MAX_DATA_SIZE; i++) {
            for (int j = 0; j < MAX_DATA_SIZE; j++) {
                data_out_buffer[ko][i][j] = 0;
            }
        }
    }

    // 分组卷积
    for (int g = 0; g < groups; g++) {
        // 输出空间分块
        for (int i = 0; i < num_blocks_h; i++) {
            int h_block_start = i * MAX_DATA_SIZE;
            int h_block_size = (i == num_blocks_h - 1) ? (out_size - h_block_start) : MAX_DATA_SIZE;

            for (int j = 0; j < num_blocks_w; j++) {
                int w_block_start = j * MAX_DATA_SIZE;
                int w_block_size = (j == num_blocks_w - 1) ? (out_size - w_block_start) : MAX_DATA_SIZE;

                // 计算输入块参数
                int input_start_h = h_block_start * stride - padding;
                int input_start_w = w_block_start * stride - padding;
                int input_block_height = h_block_size * stride + 2;
                int input_block_width = w_block_size * stride + 2;

                // 输出通道分块
                for (int ko = 0; ko < ch_out_sets; ko++) {
                    int ko_start = ko * MAX_WEIGHTS_CH_OUT;
                    int ko_size = (ko == ch_out_sets - 1) ? (ch_out_group - ko_start) : MAX_WEIGHTS_CH_OUT;

                    // 清空当前输出块缓冲区
                    for (int kk = 0; kk < ko_size; kk++) {
                        for (int ii = 0; ii < h_block_size; ii++) {
                            for (int jj = 0; jj < w_block_size; jj++) {
                                data_out_buffer[kk][ii][jj] = 0;
                            }
                        }
                    }

                    // 输入通道分块
                    for (int ki = 0; ki < ch_in_sets; ki++) {
                        int ki_start = ki * MAX_WEIGHTS_CH_IN;
                        int ki_size = (ki == ch_in_sets - 1) ? (ch_in_group - ki_start) : MAX_WEIGHTS_CH_IN;

                        // 加载输入数据（带边界处理）
                        for (int ii = 0; ii < input_block_height; ii++) {
                            int h_index = input_start_h + ii;
                            for (int jj = 0; jj < input_block_width; jj++) {
                                int w_index = input_start_w + jj;
                                for (int c = 0; c < ki_size; c++) {
                                    int global_c = g * ch_in_group + ki_start + c;
                                    if (h_index < 0 || h_index >= size || w_index < 0 || w_index >= size) {
                                        data_in_buffer[c][ii][jj] = 0;
                                    } else {
                                        data_in_buffer[c][ii][jj] = data_in[global_c * size * size + h_index * size + w_index];
                                    }
                                }
                            }
                        }

                        // 加载权重
                        for (int kk = 0; kk < ko_size; kk++) {
                            int global_ko = g * ch_out_group + ko_start + kk;
                            for (int c = 0; c < ki_size; c++) {
                                int global_ki = g * ch_in_group + ki_start + c;
                                for (int y = 0; y < 3; y++) {
                                    for (int x = 0; x < 3; x++) {
                                        weights_buffer[kk][c][y][x] = weights[global_ko * ch_in * 9 + global_ki * 9 + y * 3 + x];
                                    }
                                }
                            }
                        }

                        // 卷积计算核心
                        for (int ii = 0; ii < h_block_size; ii++) {
                            for (int jj = 0; jj < w_block_size; jj++) {
                                for (int kk = 0; kk < ko_size; kk++) {
                                    for (int y = 0; y < 3; y++) {
                                        for (int x = 0; x < 3; x++) {
#pragma HLS PIPELINE II=1
                                            int input_ii = ii * stride + y;
                                            int input_jj = jj * stride + x;
                                            for (int c = 0; c < ki_size; c++) {
                                                D32 data_val = data_in_buffer[c][input_ii][input_jj];
                                                D32 weight_val = weights_buffer[kk][c][y][x];
                                                data_out_buffer[kk][ii][jj] += data_val * weight_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } // ki循环结束

                    // 加偏置并写回结果
                    for (int kk = 0; kk < ko_size; kk++) {
                        int global_ko = g * ch_out_group + ko_start + kk;
                        D32 bias_val = bias_buffer[global_ko];
                        for (int ii = 0; ii < h_block_size; ii++) {
                            for (int jj = 0; jj < w_block_size; jj++) {
                                D32 result = data_out_buffer[kk][ii][jj] + bias_val;
                                int output_index = global_ko * out_size * out_size + 
                                                (h_block_start + ii) * out_size + 
                                                (w_block_start + jj);
                                data_out[output_index] = result;
                            }
                        }
                    }
                } // ko循环结束
            } // w分块结束
        } // h分块结束
    } // 分组循环结束
}