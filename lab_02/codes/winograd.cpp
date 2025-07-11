#include <iostream>
#include <vector>
#include <cstring>
#include <random>
#include <sys/time.h>
#include <cassert>
#include <cmath>
#include <iomanip>
const int IN_CH = 3;
const int IN_H = 56;
const int IN_W = 56;
const int OUT_CH = 64;
const int K_SIZE = 3;
const int STRIDE = 1;
const int PADDING = 0;
const int OUT_H = (IN_H + 2 * PADDING - K_SIZE) / STRIDE + 1; // 54
const int OUT_W = (IN_W + 2 * PADDING - K_SIZE) / STRIDE + 1; // 54
const int M = 2; 
const int R = 3; 
const int ALPHA = M + R - 1; 
float g_input[IN_CH][IN_H][IN_W];
float g_filter[OUT_CH][IN_CH][K_SIZE][K_SIZE];
float g_output_winograd[OUT_CH][OUT_H][OUT_W];
float g_output_im2col[OUT_CH][OUT_H][OUT_W];
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}
void init_data() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-1.0, 1.0);
    for (int c = 0; c < IN_CH; ++c)
        for (int h = 0; h < IN_H; ++h)
            for (int w = 0; w < IN_W; ++w)
                g_input[c][h][w] = distrib(gen);
    for (int n = 0; n < OUT_CH; ++n)
        for (int c = 0; c < IN_CH; ++c)
            for (int h = 0; h < K_SIZE; ++h)
                for (int w = 0; w < K_SIZE; ++w)
                    g_filter[n][c][h][w] = distrib(gen);
}
void verify_results() {
    const float tolerance = 1e-4;
    for (int n = 0; n < OUT_CH; ++n) {
        for (int h = 0; h < OUT_H; ++h) {
            for (int w = 0; w < OUT_W; ++w) {
                float diff = std::abs(g_output_winograd[n][h][w] - g_output_im2col[n][h][w]);
                assert(diff < tolerance);
            }
        }
    }
    std::cout << "Verification successful: Results are consistent." << std::endl;
}
float g_col[OUT_H * OUT_W][IN_CH * K_SIZE * K_SIZE];
float g_unfolded_filter[IN_CH * K_SIZE * K_SIZE][OUT_CH];
void im2col_convolution() {
    int col_row_idx = 0;
    for (int h_out = 0; h_out < OUT_H; ++h_out) {
        for (int w_out = 0; w_out < OUT_W; ++w_out) {
            int col_col_idx = 0;
            for (int c_in = 0; c_in < IN_CH; ++c_in) {
                for (int kh = 0; kh < K_SIZE; ++kh) {
                    for (int kw = 0; kw < K_SIZE; ++kw) {
                        g_col[col_row_idx][col_col_idx++] = g_input[c_in][h_out + kh][w_out + kw];
                    }
                }
            }
            col_row_idx++;
        }
    }
    int filter_row_idx = 0;
    for (int c_in = 0; c_in < IN_CH; ++c_in) {
        for (int kh = 0; kh < K_SIZE; ++kh) {
            for (int kw = 0; kw < K_SIZE; ++kw) {
                for (int c_out = 0; c_out < OUT_CH; ++c_out) {
                    g_unfolded_filter[filter_row_idx][c_out] = g_filter[c_out][c_in][kh][kw];
                }
                filter_row_idx++;
            }
        }
    }
    float temp_output[OUT_H * OUT_W][OUT_CH];
    memset(temp_output, 0, sizeof(temp_output));
    for (int i = 0; i < OUT_H * OUT_W; ++i) {
        for (int j = 0; j < OUT_CH; ++j) {
            for (int k = 0; k < IN_CH * K_SIZE * K_SIZE; ++k) {
                temp_output[i][j] += g_col[i][k] * g_unfolded_filter[k][j];
            }
        }
    }
    for (int c_out = 0; c_out < OUT_CH; ++c_out) {
        for (int h_out = 0; h_out < OUT_H; ++h_out) {
            for (int w_out = 0; w_out < OUT_W; ++w_out) {
                g_output_im2col[c_out][h_out][w_out] = temp_output[h_out * OUT_W + w_out][c_out];
            }
        }
    }
}
const float G[ALPHA][R] = {{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}};
const float GT[R][ALPHA] = {{1.0f, 0.5f, 0.5f, 0.0f}, {0.0f, 0.5f, -0.5f, 0.0f}, {0.0f, 0.5f, 0.5f, 1.0f}};
const float AT[M][ALPHA] = {{1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, -1.0f, -1.0f}};
const float A[ALPHA][M] = {{1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, -1.0f}, {0.0f, -1.0f}};
const float BT[ALPHA][ALPHA] = {{1.0f, 0.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, -1.0f}};
const float B[ALPHA][ALPHA] = {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, -1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, -1.0f}};
void winograd_convolution() {
    const int num_tiles_h = OUT_H / M;
    const int num_tiles_w = OUT_W / M;
    float U[OUT_CH][IN_CH][ALPHA][ALPHA];
    for (int oc = 0; oc < OUT_CH; ++oc) {
        for (int ic = 0; ic < IN_CH; ++ic) {
            float temp[ALPHA][R];
            for(int i = 0; i < ALPHA; ++i) {
                for(int j = 0; j < R; ++j) {
                    temp[i][j] = G[i][0] * g_filter[oc][ic][0][j] +
                                 G[i][1] * g_filter[oc][ic][1][j] +
                                 G[i][2] * g_filter[oc][ic][2][j];
                }
            }
            for(int i = 0; i < ALPHA; ++i) {
                for(int j = 0; j < ALPHA; ++j) {
                    U[oc][ic][i][j] = temp[i][0] * GT[0][j] +
                                      temp[i][1] * GT[1][j] +
                                      temp[i][2] * GT[2][j];
                }
            }
        }
    }
    for (int tile_h = 0; tile_h < num_tiles_h; ++tile_h) {
        for (int tile_w = 0; tile_w < num_tiles_w; ++tile_w) {
            for (int oc = 0; oc < OUT_CH; ++oc) {
                float M_sum[ALPHA][ALPHA] = {{0.0f}}; 
                for (int ic = 0; ic < IN_CH; ++ic) {
                    float d[ALPHA][ALPHA]; 
                    float V[ALPHA][ALPHA]; 
                    float temp_V[ALPHA][ALPHA];
                    int h_start = tile_h * M;
                    int w_start = tile_w * M;
                    for(int i=0; i<ALPHA; ++i)
                        for(int j=0; j<ALPHA; ++j)
                            d[i][j] = g_input[ic][h_start+i][w_start+j];
                    for(int i=0; i<ALPHA; ++i) {
                        for(int j=0; j<ALPHA; ++j) {
                            temp_V[i][j] = BT[i][0]*d[0][j] + BT[i][1]*d[1][j] + BT[i][2]*d[2][j] + BT[i][3]*d[3][j];
                        }
                    }
                     for(int i=0; i<ALPHA; ++i) {
                        for(int j=0; j<ALPHA; ++j) {
                            V[i][j] = temp_V[i][0]*B[0][j] + temp_V[i][1]*B[1][j] + temp_V[i][2]*B[2][j] + temp_V[i][3]*B[3][j];
                        }
                    }
                    for(int i=0; i<ALPHA; ++i)
                        for(int j=0; j<ALPHA; ++j)
                            M_sum[i][j] += U[oc][ic][i][j] * V[i][j];
                }
                float Y_temp[M][ALPHA];
                float Y[M][M]; 
                for(int i=0; i<M; ++i) {
                    for(int j=0; j<ALPHA; ++j) {
                        Y_temp[i][j] = AT[i][0]*M_sum[0][j] + AT[i][1]*M_sum[1][j] + AT[i][2]*M_sum[2][j] + AT[i][3]*M_sum[3][j];
                    }
                }
                for(int i=0; i<M; ++i) {
                    for(int j=0; j<M; ++j) {
                        Y[i][j] = Y_temp[i][0]*A[0][j] + Y_temp[i][1]*A[1][j] + Y_temp[i][2]*A[2][j] + Y_temp[i][3]*A[3][j];
                    }
                }
                int h_out_start = tile_h * M;
                int w_out_start = tile_w * M;
                for(int i=0; i<M; ++i)
                    for(int j=0; j<M; ++j)
                        g_output_winograd[oc][h_out_start+i][w_out_start+j] = Y[i][j];
            }
        }
    }
}
int main() {
    std::cout << "Initializing data..." << std::endl;
    init_data();
    std::cout << "\nRunning Im2col + GEMM convolution..." << std::endl;
    double time_start_im2col = get_time();
    im2col_convolution();
    double time_end_im2col = get_time();
    double im2col_duration = (time_end_im2col - time_start_im2col) * 1000.0;
    std::cout << "Im2col + GEMM finished." << std::endl;
    std::cout << "\nRunning Winograd F(2x2, 3x3) convolution..." << std::endl;
    double time_start_wino = get_time();
    winograd_convolution();
    double time_end_wino = get_time();
    double winograd_duration = (time_end_wino - time_start_wino) * 1000.0;
    std::cout << "Winograd finished." << std::endl;
    std::cout << "\n--- Verification ---" << std::endl;
    verify_results();
    std::cout << "\n--- Performance Results ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Im2col Time:   " << im2col_duration << " ms" << std::endl;
    std::cout << "Winograd Time: " << winograd_duration << " ms" << std::endl;
    return 0;
}
