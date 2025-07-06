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