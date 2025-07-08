#include<iostream>

int col[2916][27];

int (*im2col(const int X_in[3][56][56]))[27]
{

    for(int i_1 = 0; i_1 < 54; i_1 ++)
    {
        int bubu = i_1 * 54;
        for(int j_1 = 0; j_1 < 54; j_1 ++)
        {
            for(int i = 0; i < 3; i++)
            {
                int nn = i * 3;
                for(int j = 0; j < 3; j++)
                {
                    int n1 = nn * 3;
                    int n2 = j * 3;
                    for(int k = 0; k < 3; k++)
                    {
                        col[bubu + j_1][n1 + n2 + k] = X_in[i][i_1 + j][j_1 + k]; 
                    }   
                }
            }
        }
    }
    return col;
}


int main() {
    int X_input[3][56][56]; 

    // 填充一些测试数据，方便验证
    for(int c = 0; c < 3; ++c) {
        for(int h = 0; h < 56; ++h) {
            for(int w = 0; w < 56; ++w) {
                // 确保数据有区分度，方便调试
                X_input[c][h][w] = c * 1000000 + h * 1000 + w; 
            }
        }
    }

    // 调用 im2col
    int (*result_col)[27] = im2col(X_input);

    // 打印 col 矩阵的部分内容进行验证
    // 验证第一个窗口（对应 X_input[c][0:2][0:2] 的数据）
    std::cout << "验证第一个滑动窗口的数据 (col[0]的前27个元素):" << std::endl;
    for (int j = 0; j < 27; ++j) {
        std::cout << result_col[0][j] << " ";
        if ((j + 1) % (3 * 3) == 0) { // 每9个元素（一个通道的窗口）换行
            std::cout << " | ";
        }
    }
    std::cout << std::endl << std::endl;

    // 验证第二个滑动窗口（如果 stride=1，这会是X_input[c][0:2][1:3]的数据）
    std::cout << "验证第二个滑动窗口的数据 (col[1]的前27个元素):" << std::endl;
    for (int j = 0; j < 27; ++j) {
        std::cout << result_col[1][j] << " ";
        if ((j + 1) % (3 * 3) == 0) {
            std::cout << " | ";
        }
    }
    std::cout << std::endl;

    // 可以手动计算 X_input 中的一些值来对比验证 col 数组的正确性
    // 例如，col[0][0] 应该等于 X_input[0][0][0]
    // col[0][8] 应该等于 X_input[0][2][2]
    // col[0][9] 应该等于 X_input[1][0][0] 等

    return 0;
}