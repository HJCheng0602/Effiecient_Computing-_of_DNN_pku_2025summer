#include <iostream>
#include <cstring>
#include <string.h>
using namespace std;

int col[2916][27];
int unfolded_filter[27][64];
int unfolded_filter_T[64][27];

int temp_output[2916][64];

int output[64][54][54];

int (*im2col(const int X_in[3][56][56]))[27]
{

    for (int i_1 = 0; i_1 < 54; i_1++)
    {
        int bubu = i_1 * 54;
        for (int j_1 = 0; j_1 < 54; j_1++)
        {
            for (int i = 0; i < 3; i++)
            {
                int nn = i * 3;
                for (int j = 0; j < 3; j++)
                {
                    int n1 = nn * 3;
                    int n2 = j * 3;
                    for (int k = 0; k < 3; k++)
                    {
                        col[bubu + j_1][n1 + n2 + k] = X_in[i][i_1 + j][j_1 + k];
                    }
                }
            }
        }
    }
    return col;
}

int (*unfold_filter(const int filter[64][3][3][3]))[64]
{
    for(int i = 0; i < 64; i++)
    {
        for(int j1 = 0; j1 < 3; j1++)
        {
            int temp1 = j1 * 3;
            for(int j2 = 0; j2 < 3; j2 ++)
            {
                int temp2 = j2 * 3;
                int temp3 = temp1 * 3;
                for(int j3 = 0; j3 < 3; j3 ++)
                {
                    unfolded_filter[temp3 + temp2 + j3][i] = filter[i][j1][j2][j3];
                }
            }
        }
    }
    return unfolded_filter;
}

void matmul_correct()
{
    memset(temp_output, 0, sizeof(temp_output));
    for (int i = 0; i < 2916; i++) 
    {
        for (int j = 0; j < 64; j++) 
        {
            for (int k = 0; k < 27; k++) 
            {
                
                temp_output[i][j] += col[i][k] * unfolded_filter[k][j]; 
            }
        }
    }
}


void set_matrix_style()
{
    for(int i = 0; i < 64; i++)
    {
        for(int j = 0; j < 54; j++)
        {
            int temp1 = j * 54;
            for(int k = 0; k < 54; k++)
            {
                output[i][j][k] = temp_output[temp1 + k][i];
            }
        }
    }
}

int main()
{
    
    
}