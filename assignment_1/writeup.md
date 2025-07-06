# 第一次大作业


## Q1
### (a)
先进行线性操作：
$$
y_1 = x_1w_1 + x_2w_2+x_3w_3
$$
然后再进入$sigmoid$层：
$$
y = \frac{1}{1 + e^{-(x_1w_1 + x_2w_2 + x_3w_3)}}
$$
### (b)
首先，我们将$loss$对$y$求导：
$$
\frac{\partial loss}{\partial y} = 2(y - t)
$$
然后根据链式法则，不难得出：
$$
\frac{\partial loss}{\partial w_i} = 2(\frac{1}{1 + e^{-(x_1w_1+x_2w_2+x_3w_3)}}-t)\cdot\frac{x_ie^{-(x_1w_1+x_2w_2+x_3w_3)}}{(1 + e^{-(x_1w_1+x_2w_2+x_3w_3)})^2}\\i = 1,2,3
$$
### (c)
问题：

很容易我们就能发现$sigmoid$函数在输入极大或极小的时候会发生梯度消失现象；然后根据链式法则，当与DNN结合是，如果网络很深且$sigmoid$函数的输出原因使得输出接近$0$或$1$，使得结果极小，导致靠近输出层的权重梯度极其小，即发生了梯度消失现象。

解决：
- 使用Relu函数避免梯度消失现象
- 使用resnet允许梯度绕过一层或多层直接向下传输，避免了网络深度过深导致的梯度消失现象。
- 使用BN层，避免导数过小的问题。


## Q2
### (a)
考虑$J(x) + 4x^2$的函数表达式如下：
$$
\Gamma(x) = (x - 2)^2 + 4x^2 = 5x^2 -4x + 4
$$
取$x$的对称轴$x_0 = \frac{2}{5}$
代入方程得：
$$
\Gamma(x_0) = 5 \times\frac{4}{25}-4 \times \frac{2}{5} + 4 = \frac{16}{5}
$$
显然该函数的全局最小值为$\frac{16}{5}$，在$x = \frac{2}{5}$时取得。
### (b)
我们需要进行分类讨论:
- $x \geq 0$时，原式为：
    $$
    \Gamma_1(x) = x^2 + 4
    $$
    显然其最小值在$x = 0$处取得，最小值为4
- $x < 0$时，原式为：
    $$
    \Gamma_2(x) = x^2 - 8x + 4
    $$
    显然其在$x < 0$上大于4

因此该函数的全局最小值为4，在$x = 0$时取得。
### (c)
还是需要进行分类讨论：
- 当$x \geq 0$时，原式为：
    $$
    \Gamma_1(x) = x^2 + (\alpha - 4)x + 4
    $$
    还是需要进行分类讨论：
    - 当$\alpha \geq 4$时，$[0, +\infin)$上的最小值为$4$，在$x_0 = 0$时取得。
    - 当$\alpha < 4$时，最小值不在 x = 0处。
- 考虑当$x < 0$时，原式为：
    $$
    \Gamma_2(x) = x^2 + (-\alpha - 4)x + 4
    $$
    显然，通过分析，我们发现当$\alpha \geq -4$时，最小值不在其他处取得

因此，综上所述，当$\alpha \geq 4$时，最小值点在$x = 0$处取得。

### (d)
我们发现在$l_2$正则化中，原函数的最小值点被从$2$拉到了$\frac{2}{5}$，说明$l_2$正则化会惩罚较大的权重值，促使模型参数更小，但是其力度通常不够大，但是这也反映了其相对平滑，而且通过对其数学表达式的分析，显而易见其只能使参数尽可能地接近$0$，而非直接让参数变成0；

然而在$l_1$正则化中，当正则化参数足够大时，我们可以发现参数被强制变成了$0$，这说明$|x|$项在$x = 0$时的导数不确定的特性使得其能将不重要的特征全中压缩到0，这意味着其可以直接进行特征选择，把一些特征对应的权重设为$0$，从而将这些特征从模型中移除，从而达到稀疏性的目的。

因此，$l_1$正则化更能促进参数的稀疏性。

## Q3
### (a)
通过简单计算不难得出：
$$
P = Q = 3
$$

### (b)
直接卷积的代码如下：
```Cpp
#include <iostream>
#include <vector>

using namespace std;

class Concolution
{
private:
    int K = 0;
    int C = 0;
    int R = 0;
    int S = 0;
    vector<vector<vector<vector<int>>>> filters;
public:
    Concolution()
    {
    }
    void set_filters(vector<vector<vector<vector<int>>>> a)
    {
        filters = a;
    }
    vector<vector<vector<int>>> Con_calculate(vector<vector<vector<int>>> input)
    {
        int H = input[0].size();
        int W = input[0][0].size();
        vector<vector<vector<int>>> out_put(K, vector<vector<int>>(H - (R - 1), vector<int>(W - (S - 1), 0)));
        for (int o = 0; o < K; o++)
        {
            for (int i1 = 0; i1 < H - (R - 1); i1++)
            {
                for (int j1 = 0; j1 < W - (S - 1); j1++)
                {
                    int sum = 0;
                    for (int q = 0; q < C; q++)
                    {
                        for (int i = 0; i < R; i++)
                        {
                            for (int j = 0; j < S; j++)
                            {
                                sum += input[q][i + i1][j + j1] * filters[o][q][i][j];
                            }
                        }
                    }
                    out_put[o][i1][j1] = sum;
                }
            }
        }
        return out_put;
    }
};
```

### (c)
- 输入通道级别上的循环展开：
    ```Cpp
    #include <iostream>
    #include <vector>

    using namespace std;

    class Concolution
    {
    private:
        int K = 0;
        int C = 0;
        int R = 0;
        int S = 0;
        vector<vector<vector<vector<int>>>> filters;
    public:
        Concolution()
        {
        }
        void set_filters(vector<vector<vector<vector<int>>>> a)
        {
            filters = a;
        }
        vector<vector<vector<int>>> Con_calculate_InputChannelUnrolled(const vector<vector<vector<int>>>& input)
        {

            int H = input[0].size();
            int W = input[0][0].size();
            int P = H - R + 1;
            int Q = W - S + 1;
            vector<vector<vector<int>>> out_put(K, vector<vector<int>>(P, vector<int>(Q, 0)));
            for (int k_idx = 0; k_idx < K;k_idx++)
            {
                for (int p_idx = 0; p_idx < P; p_idx++)
                {
                    for (int q_idx = 0; q_idx < Q; q_idx++)
                    {
                        int sum = 0;
                        // 主循环：以 2 为步长进行展开计算
                        int c_idx_unrolled_limit = C_in - (C_in % 2); // 确保是偶数
                        for (int c_idx = 0; c_idx < c_idx_unrolled_limit; c_idx += 2)
                        {
                            int c1_in_idx = c_idx;
                            int c2_in_idx = c_idx + 1;

                            for (int r_idx = 0; r_idx < R_k; ++r_idx)
                            {
                                for (int s_idx = 0; s_idx < S_k; ++s_idx)
                                {
                                    // 处理第一个输入通道
                                    sum += input[c1_in_idx][p_idx + r_idx][q_idx + s_idx] *
                                        filters[k_idx][c1_in_idx][r_idx][s_idx];
                                    // 处理第二个输入通道
                                    sum += input[c2_in_idx][p_idx + r_idx][q_idx + s_idx] *
                                        filters[k_idx][c2_in_idx][r_idx][s_idx];
                                }
                            }
                        }

                        // 尾部循环：处理剩余的（可能是 1 个）通道
                        for (int c_idx = c_idx_unrolled_limit; c_idx < C_in; ++c_idx)
                        {
                            for (int r_idx = 0; r_idx < R_k; ++r_idx)
                            {
                                for (int s_idx = 0; s_idx < S_k; ++s_idx)
                                {
                                    sum += input[c_idx][p_idx + r_idx][q_idx + s_idx] *
                                        filters[k_idx][c_idx][r_idx][s_idx];
                                }
                            }
                        }
                        out_put[k_idx][p_idx][q_idx] = sum;
                    }
                }
            }
            return out_put;
        }
    };
    ```
- 输出通道级别的循环展开
    ```cpp
    #include <iostream>
    #include <vector>

    using namespace std;

    class Concolution
    {
    private:
        int K = 0;
        int C = 0;
        int R = 0;
        int S = 0;
        vector<vector<vector<vector<int>>>> filters;
    public:
        Concolution()
        {
        }
        void set_filters(vector<vector<vector<vector<int>>>> a)
        {
            filters = a;
        }
        vector<vector<vector<int>>> Con_calculate_InputChannelUnrolled(const vector<vector<vector<int>>>& input)
        {

            int H = input[0].size();
            int W = input[0][0].size();
            int P = H - R + 1;
            int Q = W - S + 1;
            vector<vector<vector<int>>> out_put(K, vector<vector<int>>(P, vector<int>(Q, 0)));
            // 循环展开输入通道 K。这里我们展开 2 次。
            int k_idx_unrolled_limit = K - (K % 2);
            for (int k_idx = 0; k_idx < k_idx_unrolled_limit;k_idx+=2)
            {
                for (int p_idx = 0; p_idx < P; p_idx++)
                {
                    for (int q_idx = 0; q_idx < Q; q_idx++)
                    {
                        int k1_idx = k_idx;
                        int k2_idx = k_idx + 1;

                        int sum1 = 0;
                        int sum2 = 0;
                        
                        for (int c_idx = 0; c_idx < C; c_idx ++)
                        {
                            

                            for (int r_idx = 0; r_idx < R; r_idx++)
                            {
                                for (int s_idx = 0; s_idx < S; s_idx ++)
                                {
                                    // 处理第一个通道
                                    sum1 += input[c_idx][p_idx + r_idx][q_idx + s_idx] *
                                        filters[k1_idx][c_idx][r_idx][s_idx];
                                    // 处理第二个通道
                                    sum2 += input[c_idx][p_idx + r_idx][q_idx + s_idx] *
                                        filters[k2_idx][c_idx][r_idx][s_idx];
                                }
                            }
                        }
                        out_put[k1_idx][p_idx][q_idx] = sum1;
                        out_put[k2_idx][p_idx][q_idx] = sum2;
                    }
                }
            }
            for(int k_idx = k_idx_unrolled_limit; k_idx < K; k_idx++)
            {
                for(int p_idx = 0; p_idx < P; p_idx ++)
                {
                    for(int q_idx = 0; q_idx < Q; q_idx ++)
                    {
                        int sum = 0;
                        for (int c_idx = 0; c_idx < C; c_idx ++)
                        {
                            for (int r_idx = 0; r_idx < R; r_idx++)
                            {
                                for (int s_idx = 0; s_idx < S; s_idx ++)
                                {
                                    sum += input[c_idx][p_idx + r_idx][q_idx + s_idx] *
                                        filters[k_idx][c_idx][r_idx][s_idx];
                                    
                                }
                            }
                        }
                        out_put[k_idx][p_idx][q_idx] = sum;
                        
                    }
                }
            }
            return out_put;
        }
    };
    ```

## Q4
### (a)
首先我们构建输入hash map：

| 0|    (0,1)     |
|---|---|
| 1    | (1,2)        |
| 2 | (2,3)    |

### (b)
先对$P_1$构建输出哈希表：

|(0,1)|
|---|
(0,0)|

然后再为$P_2$构建哈希表：
|(0,0)|
|---|
(1,0)|
(0,1)|
(1,1)|
(0,2)|
(1,2)|

然后我们对$P_3$构建输出哈希表：

|(0,1)|
|---|
(1,1)|
(2,1)|
(0,2)|
|(1,2)|
(2,2)|

然后，我们尝试合并输出哈希表：
|||
|---|---|
|0|(0,0)
|1|(0,1)
|2|(0,2)
|3|(1,0)
|4|(1,1)
|5|(1,2)
|6|(2,1)
|7|(2,2)

综上，上面的表格即为输出哈希表

### (c)
我们来根据(a)(b)来构建规则手册：


对于$P_1$

|(0,1)|$\to$|(-1,-1)|
|---|---|---|
(0,0)|$\to$|(-1, 0)|

对于$P_2$
|(0,0)|$\to$|(0,1)|
|---|---|---|
(1,0)|$\to$|(-1,1)|
(0,1)|$\to$|(0,0)|
(1,1)|$\to$|(-1,0)|
(0,2)|$\to$|(0,-1)|
(1,2)|$\to$|(-1,-1)|

对于$P_3$
|(0,1)|$\to$|(1,1)|
|---|---|---|
(1,1)|$\to$|(0,1)|
(2,1)|$\to$|(-1,1)|
(0,2)|$\to$|(1,0)|
|(1,2)|$\to$|(0,0)|
(2,2)|$\to$|(-1,0)|


那么我们先定义这样的映射规则：
$$
P_1 \to num[0]\\
P_2 \to num[1]\\
P_3 \to num[2]  
$$

那么我们便可以编写我们的rulebook如下：
|Offset|count|in|out|
|---|---|---|:---:|
(-1,-1)|0|0|1|
||1|1|5|
|(0,-1)|0|1|2|
|(-1,0)|0|0|0|
||1|1|4|
||2|2|7|
|(0,0)|0|1|1|
||1|2|5|
|(1,0)|0|2|2|
|(-1,1)|0|1|3|
||1|2|6|
|(0,1)|0|1|0|
||1|2|4|
|(1,1)|0|2|1|


### (d)
所有具有相同行output的行对应的乘积结果将都会被累加到该output所指示的一个输出位置上。

举例说明：
我们设输入为：

`Input[0,0,0] = 5`

`Input[0,1,0] = 3`

`Input[1,0,0] = 2`

`其余为 0。`

然后不妨设卷积和是一个这样的卷积和：
```
W_0 = [[1, 2],
       [3, 4]]
```

那么我们就很容易的得出了我们的输入哈希表：
| in_idx | (h, w, c) | Value |
|--------|-----------|-------|
| 0      | (0, 0, 0) | 5     |
| 1      | (0, 1, 0) | 3     |
| 2      | (1, 0, 0) | 2     |

然后我们现在只关注`output[0,,0,0]`
| out_idx | (p, q, k) | Current_Value |
|---------|-----------|---------------|
| 0       | (0, 0, 0) | 0 (初始化为0) |

那么对应的规则手册就是：
| rule_id | in_idx | kernel_weight_coords (k, c, r, s) | out_idx |
|---------|--------|-----------------------------------|---------|
| 0       | 0      | (0, 0, 0, 0)                      | 0       |
| 1       | 1      | (0, 0, 0, 1)                      | 0       |
| 2       | 2      | (0, 0, 1, 0)                      | 0       |

```cpp
// 假设这是稀疏卷积引擎的伪代码
// output_hash_table[out_idx] 存储输出索引对应的值

// 初始化 output_hash_table[0].Current_Value = 0

// 处理 rule_id = 0 的规则：
// out_idx = 0
// input_value = InputHash[0].Value = 5
// weight_value = W[0][0][0][0] = 1
// product = 5 * 1 = 5
// output_hash_table[0].Current_Value += product; // output_hash_table[0].Current_Value 现在是 0 + 5 = 5

// 处理 rule_id = 1 的规则：
// out_idx = 0
// input_value = InputHash[1].Value = 3
// weight_value = W[0][0][0][1] = 2
// product = 3 * 2 = 6
// output_hash_table[0].Current_Value += product; // output_hash_table[0].Current_Value 现在是 5 + 6 = 11

// 处理 rule_id = 2 的规则：
// out_idx = 0
// input_value = InputHash[2].Value = 2
// weight_value = W[0][0][1][0] = 3
// product = 2 * 3 = 6
// output_hash_table[0].Current_Value += product; // output_hash_table[0].Current_Value 现在是 11 + 6 = 17

// 所有相关规则处理完毕后，Output[0,0,0] 的最终值为 17。
```


因此，当规则手册中的多行具有相同的输出索引（out_idx）时，在稀疏卷积结果计算过程中，所有这些行对应的输入值与权重值的乘积都会被累加到该共享的输出位置上。这正是卷积操作中“求和”部分的实现方式，通过这种机制，稀疏卷积能够正确地计算出输出特征图的非零元素值。这种设计是稀疏卷积高效性的关键，因为它只处理那些实际有贡献的乘积和累加，而忽略大量的零值操作。