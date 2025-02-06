# Report

|           | V0        | V1        |
| --------- | --------- | --------- |
| Times(s)  | 28.8741   | 0.0729225 |
| GFLOPS    | 0.0569426 | 22.5468   |
| LUT       | 8947      | 25716     |
| LUTAsMem  | 695       | 728       |
| REG       | 9378      | 39736     |
| BRAM      | 79        | 79        |
| URAM      | 407       | 467       |
| DSP       | 7         | 322       |
| Freq(MHz) | 200       | 200       |
| WNS(ns)   | 0.035     | 0.057     |

## Performance  Analysis

From cnn.h we get,
$$
kNum = 64
$$

$$
kKernel = 4
$$

$$
kInImSize = 116
$$

$$
kOutImSize = 112
$$

Total Flops can be calculated by,
$$
\text{FLOPS} = kOutImSize \times kOutImSize \times kNum \times kNum \times (kKernel \times kKernel)
$$
Therefore,
$$
FLOPS = 882,083,584
$$
If II=1, it means that, under ideal conditions, each operation in the innermost loop can be initiated every clock cycle. When running on 200MHz,
$$
Estimated Cycle = (882,083,584)/(200,000,000) = 4.11s
$$


### V0

With no optimization at all, from v++.log, we know II=7. Therefore, the estimated running time is 7*4.11 = 28.77s.

### V1

After changing the loop order of convolution into this,

```c++
    for (int h = 0; h < kOutImSize; ++h) {
        for(int j = 0; j < kNum; ++j){
            for (int p = 0; p < kKernel; ++p) {
                for (int q = 0; q < kKernel; ++q){
                    for (int w = 0; w < kOutImSize; ++w) {
#pragma HLS PIPELINE II=1
                        for (int i = 0; i < kNum; ++i) {
                            local_output[h][w][i] += local_input[h+p][w+q][j] * local_weight[p][q][i][j];
                        }
                    }
				}
			}
		}
	}
```

The estimated speed-up is 64x. Therefore, the estimated running time is 4.11/64 = 0.064s.

To make sure the II=1, the following optimization methods are used.

- Change `local_output` from `RAM_1P_URAM` to `RAM_2P_URAM`, i.e., dual-port URAM. This change is crucial because, in the innermost loop, there are read-modify-write operations on the same data unit. With dual-port memory, one port is used for reading and the other for writing, greatly reducing read-write conflicts and scheduling difficulty, thus increasing the likelihood of achieving **II=1**.
- Perform **complete partitioning** on the `i` dimension (channel dimension) of `local_output` (`#pragma HLS ARRAY_PARTITION complete dim=3`). This means that for every spatial position (h,w)(h, w), all elements in the `i` dimension of `local_output[h][w][i]` are split into independent storage units, enabling simultaneous access to different channels in the same cycle.
- Perform **complete partitioning** on the `i` dimension  of `local_weight` (`#pragma HLS ARRAY_PARTITION variable=local_weight complete dim=3`). By completely partitioning the weights of the kernel across the channel dimension, multiple channel weight data can be accessed simultaneously in the innermost loop. This removes the limitation of a single memory port, accelerating access and reducing bandwidth constraints.
- Use the `#pragma HLS PIPELINE II=1` directive in the innermost loop to instruct the tool to initiate a new iteration every clock cycle, thereby achieving **instruction-level parallelism (ILP)**. By combining array partitioning and dual-port memory, the tool has the opportunity to complete multiple data accesses and computations within a single clock cycle, striving to achieve the performance target of **II=1**.

## Chip Layout

### v0

![v0](v0-17343882324641.png)

### v1

![v1](v1.png)

