/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.
...【版权声明省略】...
**********/

#include "hls_stream.h"
#include "ap_int.h"
#include "cnn.h"

extern "C" {

void cnn(DTYPE *input,  DTYPE *weight, DTYPE *output)
{
#pragma HLS INTERFACE m_axi port = input offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = input bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    DTYPE local_input[kInImSize][kInImSize][kNum];
#pragma HLS RESOURCE variable=local_input core=RAM_1P_URAM

    DTYPE local_output[kOutImSize][kOutImSize][kNum];
#pragma HLS RESOURCE variable=local_output core=RAM_2P_URAM
    // 对local_output在i维度complete分区
#pragma HLS ARRAY_PARTITION variable=local_output complete dim=3

    DTYPE local_weight[kKernel][kKernel][kNum][kNum];
    // 对local_weight在i,j维complete分区
#pragma HLS ARRAY_PARTITION variable=local_weight complete dim=3
//#pragma HLS ARRAY_PARTITION variable=local_weight complete dim=4

    // 从global memory读入input，不pipeline
    for (int h = 0; h < kInImSize; ++h) {
        for (int w = 0; w < kInImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                local_input[h][w][i] = input[(h*kInImSize+w)*kNum+i];
            }
        }
    }

    // 读入weight，不pipeline
    for (int p = 0; p < kKernel; ++p) {
        for (int q = 0; q < kKernel; ++q){
            for (int i = 0; i < kNum; ++i) {
                for (int j = 0; j < kNum; ++j) {
                    local_weight[p][q][i][j] = weight[((p*kKernel+q)*kNum+i)*kNum+j];
                }
            }
        }
    }

    // 初始化output为0，不pipeline
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                local_output[h][w][i] = 0.0f;
            }
        }
    }

	// Convolution计算
    // 循环顺序：h -> j -> p -> q -> w -> i
    // 在最内层循环中对i维访问local_output[h][w][i]、local_weight[p][q][i][j]以及local_input[h+p][w+q][j]
    // 由于local_weight和local_output在i和(i,j)维complete分区，多通道并行访问成为可能。
    for (int h = 0; h < kOutImSize; ++h) {
        for(int j = 0; j < kNum; ++j){
            for (int p = 0; p < kKernel; ++p) {
                for (int q = 0; q < kKernel; ++q){
                    for (int w = 0; w < kOutImSize; ++w) {
                        // 此处pipeline II=1确保在每个周期完成对i维的并行访问和加法累加
#pragma HLS PIPELINE II=1
                        for (int i = 0; i < kNum; ++i) {
                            local_output[h][w][i] += local_input[h+p][w+q][j] * local_weight[p][q][i][j];
                        }
                    }
				}
			}
		}
	}

    // 写回输出数据，不pipeline
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                output[(h*kOutImSize + w)*kNum+i] = local_output[h][w][i];
            }
        }
    }
}

}
