# ECDNN 第2次作业

## Q1

### (a)
由已知公式可得，我们先算$\bar{W}$:
$$
\bar{W} = W - mean(W) = W
$$

然后计算$u_i$

$$u_i = -1 + (i - 1) = i - 2
$$

最后计算$B_1$、$B_2$、$B_3$:
$$
B_1 = sign(\bar{W} + u_1 std(W)) = \begin{bmatrix}

    -1 & 1\\
    -1 & -1

    
\end{bmatrix}

$$

$$
B_2 = sign(\bar{W} + u_2 std(W)) = \begin{bmatrix}
    -1 & 1\\
    -1 & 1
\end{bmatrix}
$$

$$
B_3 = sign(\bar{W} + u_3 std(W)) = \begin{bmatrix}
    -1 & 1\\
    1 & 1
\end{bmatrix}
$$

以上就是$W$的三个基

### (b)
显然，我们代入$\alpha_i$和$B_i$
$$
W_{apro} = \alpha_1 B_1 + \alpha_2 B_2 + \alpha_3 B_3 = \begin{bmatrix}
    -0.13 & 0.13 \\
    -0.065 & 0.075
\end{bmatrix}
$$

## Q2

### (a)
首先我们来计算缩放因子：
$$
s = \frac{r_{max} - r_{min}}{q_{max} - q_{min}} = 2
$$ 

然后我们计算零点:

$$
z = round(q_{min} - \frac{r_{min}}{s}) = 0
$$

那么根据我们算出的数据，我们的量化函数是：
$$
W_{q,i} = round(\frac{W_i}{s}) + z = round(\frac{W_i}{2}) + 0 = round(\frac{W_i}{2})
$$

然后我们对$W$逐个量化，并组合量化结果:
$$
W_q = [-1, 1, 0, 1]
$$
### (b)
根据链式法则可以得知：
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W_q} \cdot \frac{\partial W_q}{\partial W}
$$
根据STE的近似法则：
$$
\frac{\partial W_q}{\partial W} \approxeq 2
$$

因此，梯度计算的公式就变成了：
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W_q}
$$

因此，关于原始权重$W$的梯度就是：
$$
\frac{\partial L}{\partial W} = [0.1, 0.15, 0.2, 0.25]
$$

## Q3
### (a)
首先，我们对$A$进行量化：

- 真实值范围为：-2.2 ～ 2.2
- 量化值范围：-8～7

那么我们来计算缩放因子$s$
$$
s = \frac{2.2}{8} \approxeq 0.275
$$
然后我们再来计算$Z$
$$
Z = floor(\frac{r_{min}}{s}) - q_{min} = 0
$$

因此，我们计算出了我们的量化函数：
$$
Q(r) = floor(\frac{r}{s}) - Z = floor(\frac{r}{0.275}) 
$$
因此量化后的$A$为：
$$
A_q = [-8,-4,4,7]
$$
然后我们进行反量化过程：
$$
\hat{r} = S(Q(r) + Z)
$$
将具体书据代入可知：
$$
\hat{r} = 0.275 \times r_q 
$$
因此，反量化后的向量为：
$$
\hat{A} = [-2.2, -1.1, 1.1, 2.2]
$$


然后，我们对$B$进行操作：
- 真实值范围为: 0.3~0.5
- 量化值范围: -8 ~ 7

那么我们继续来计算缩放因子：
$$
s = \frac{0.5}{8} \approxeq 0.0625
$$

然后我们再来计算$Z$:
$$
Z = floor(\frac{r_{min}}{s}) - q_{min} = floor(\frac{0.3}{0.0625}) + 8 =  28
$$

因此，我们又计算出了我们的量化函数：
$$
Q(r) = floor(\frac{r}{0.0625}) - 28
$$
因此我们量化后的$B$为：
$$
B_q = [7, -7, -7, 7]^T
$$
然后进行反量化：
$$
\hat{r} = S(Q(r) + Z)
$$
可得：
$$
\hat{B} = [0.5,0.3,0.3,0.5]^T
$$

然后我们进行计算:
$$
A_q \cdot B_q = [-8,-4,4,7] \cdot [7, -7, -7, 7]^T = -56 + 28 - 28 + 49 = -7
$$
### (b)
首先，我们对$A$进行量化：

- 真实值范围为：-2.2 ～ 2.2
- 量化值范围：-7～7

那么我们来计算缩放因子$s$
$$
s = \frac{4.4}{14} \approxeq 0.3143
$$
然后我们再来计算$Z$
$$
Z = round(\frac{r_{min}}{s}) - q_{min} = 0
$$

因此，我们计算出了我们的量化函数：
$$
Q(r) = round(\frac{r}{s}) - Z = round(\frac{r}{0.3143}) 
$$
因此量化后的$A$为：
$$
A_q = [-7,-4,4,7]
$$
然后我们进行反量化过程：
$$
\hat{r} = S(Q(r) + Z)
$$
将具体书据代入可知：
$$
\hat{r} = 0.3143 \times r_q 
$$
因此，反量化后的向量为：
$$
\hat{A} = [-2.2, -1.1, 1.1, 2.2]
$$


然后，我们对$B$进行操作：
- 真实值范围为: 0.3~0.5
- 量化值范围: -7 ~ 7

那么我们继续来计算缩放因子：
$$
s = \frac{0.5}{7} \approxeq 0.0714
$$

然后我们再来计算$Z$:
$$
Z = floor(\frac{r_{min}}{s}) - q_{min} = floor(\frac{0.3}{0.0714}) + 7 =  28
$$

因此，我们又计算出了我们的量化函数：
$$
Q(r) = floor(\frac{r}{0.0714}) - 28
$$
因此我们量化后的$B$为：
$$
B = [7, -7, -7, 7]^T
$$
然后进行反量化：
$$
\hat{r} = S(Q(r) + Z)
$$
可得：
$$
\hat{B} = [0.5,0.3,0.3,0.5]
$$

然后我们进行计算:
$$
A_q \cdot B_q = [-7,-4,4,7] \cdot [7, -7, -7, 7]^T = -49 + 28 - 28 + 49 = 0
$$

## Q4
### (a)
先计算$D_{KL}(p||q)$:
$$
D_{KL}(p||q) = \sum_{x = 1}^2P(x)\log(\frac{P(x)}{Q(x)}) = 0.2 * \log(\frac{0.2}{0.6}) + 0.8 * \log(\frac{0.8}{0.4}) = -0.2197 + 0.5545 = 0.3348
$$

然后计算$D_{KL}(q||p)$:
$$
D_{KL}(q||p) = \sum_{x = 1}^2Q(x)\log(\frac{Q(x)}{P(x)}) = 0.6 * \log3 - 0.4 * \log(2) = 0.6591 - 0.2772 = 0.3819
$$

发现KL散度的计算是不可交换的，即两者的数值并不相同

### (b)
我们有两种写法：

$$
\min_{\theta}D_{KL}(p||q_{\theta}) = \sum
_{x = 1}^2P(x)\log(\frac{P(x)}{Q_{\theta}(x)}) \\
$$
$$
\min_{\theta}D_{KL}(q_{\theta}||p) = \sum_{x = 1}^2Q_{\theta}(x)\log(\frac{Q_{\theta}(x)}{P(x)})
$$
对于第一种，我们称之为**Forward KL**，它的目标是最小化真实分布$P$和近似分布$Q_{\theta}$之间的距离。对于第二种，我们称之为**Reverse KL**，它的目标是最小化近似分布$Q_{\theta}$和真实分布$P$之间的距离。

对于第一种，我们一定不能出现：
$$
P(x) \neq 0 \land Q_{\theta}(x) = 0
$$

因此可以推导出:
$$
P(x) \neq 0 \Rightarrow Q_{\theta}(x) \neq 0
$$


对于第二种，我们就可以推导出了:
$$
P(x) = 0 \Rightarrow Q_{\theta}(x) = 0
$$

因此，Forward KL和Reverse KL的区别在于它们对分布的支持集的要求不同。Forward KL要求近似分布$Q_{\theta}$覆盖真实分布$P$的支持集，而Reverse KL则要求真实分布$P$覆盖近似分布$Q_{\theta}$的支持集。

（注：(c)的推导来自助教习题课）