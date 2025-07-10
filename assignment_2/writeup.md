# ECDNN 第2次作业

## Q1
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

## Q2
显然，我们代入$\alpha_i$和$B_i$
$$
W_{apro} = \alpha_1 B_1 + \alpha_2 B_2 + \alpha_3 B_3 = \begin{bmatrix}
    -0.13 & 0.13 \\
    -0.065 & 0.075
\end{bmatrix}
$$

