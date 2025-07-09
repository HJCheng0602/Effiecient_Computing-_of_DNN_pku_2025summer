#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <random>
#include <vector>
#include <algorithm> 
#ifdef __ARM_NEON__
#include <arm_neon.h> 
#endif
// 为了Strassen简化，这里假设I, K, J都是1024，并且为2的幂
constexpr int I = 1024;
constexpr int K = 1024;
constexpr int J = 1024;
// 缓存块大小
constexpr int BLOCK_SIZE_I = 32;
constexpr int BLOCK_SIZE_K = 32;
constexpr int BLOCK_SIZE_J = 32;
constexpr int STRASSEN_THRESHOLD = 64; 
alignas(16) int A[I][K];
alignas(16) int B[K][J];
alignas(16) int BT[J][K]; // 转置B
alignas(16) int AT[K][I]; // 转置A
alignas(16) int C[I][J];
alignas(16) int C_groundtruth[I][J];
alignas(16) int S1[I/2][J/2]; 
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}
void init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 10);
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = distrib(gen);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < J; j++) {
            B[i][j] = distrib(gen);
        }
    }
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            long long sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (long long)A[i][k] * B[k][j];
            }
            C_groundtruth[i][j] = static_cast<int>(sum);
        }
    }
}
void test() {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      assert(C[i][j] == C_groundtruth[i][j]);
    }
  }
}
// 原始的ijk顺序矩阵乘法
void matmul_ijk() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}
// 原始的ikj顺序矩阵乘法
void matmul_ikj() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < J; j++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}
// 原始的AT矩阵乘法
void matmul_AT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < I; j++) {
      AT[i][j] = A[j][i];
    }
  }
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += AT[k][i] * B[k][j];    
      }   
    }
  }
}
// 原始的BT矩阵乘法
void matmul_BT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < J; i++) {
    for (int j = 0; j < K; j++) {
      BT[i][j] = B[j][i];
    }
  }
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * BT[j][k];    
      }   
    }
  }
}
// 循环展开 (Loop Unrolling) - 以ikj顺序为例，展开最内层循环
void matmul_ikj_unrolled() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < J; j += 4) { // J假设能被4整除
        C[i][j] += A[i][k] * B[k][j];    
        C[i][j+1] += A[i][k] * B[k][j+1];    
        C[i][j+2] += A[i][k] * B[k][j+2];    
        C[i][j+3] += A[i][k] * B[k][j+3];    
      }   
    }
  }
}
// 分块/切片 (Tiling)
void matmul_tiled() {
  memset(C, 0, sizeof(C));
  for (int ii = 0; ii < I; ii += BLOCK_SIZE_I) {
    for (int jj = 0; jj < J; jj += BLOCK_SIZE_J) {
      for (int kk = 0; kk < K; kk += BLOCK_SIZE_K) {
        for (int i = ii; i < std::min(ii + BLOCK_SIZE_I, I); i++) {
          for (int j = jj; j < std::min(jj + BLOCK_SIZE_J, J); j++) {
            for (int k = kk; k < std::min(kk + BLOCK_SIZE_K, K); k++) {
              C[i][j] += A[i][k] * B[k][j];
            }
          }
        }
      }
    }
  }
}
// 写入缓存优化 (Writing Caching)
void matmul_ikj_write_optimized() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            int temp_A_ik = A[i][k];
            for (int j = 0; j < J; ++j) {
                C[i][j] += temp_A_ik * B[k][j];
            }
        }
    }
}
// 向量化 (Vectorization (SIMD)) - 使用 ARM NEON Intrinsics
void matmul_simd() {
    memset(C, 0, sizeof(C));
    constexpr int SIMD_WIDTH = 4; // 128位NEON向量包含4个32位整数

    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            int32x4_t a_val = vdupq_n_s32(A[i][k]);
            for (int j = 0; j < J; j += SIMD_WIDTH) {
                int32x4_t c_vec = vld1q_s32(C[i] + j);
                int32x4_t b_vec = vld1q_s32(B[k] + j);
                int32x4_t prod_vec = vmulq_s32(a_val, b_vec);
                c_vec = vaddq_s32(c_vec, prod_vec);
                vst1q_s32(C[i] + j, c_vec);
            }
        }
    }
}
// 数组打包 (Array packing) - 通过转置BT来优化B的访问模式
void matmul_BT_optimized() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < J; i++) {
    for (int j = 0; j < K; j++) {
      BT[i][j] = B[j][i];
    }
  }
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * BT[j][k];    
      }   
    }
  }
}
// 加法 C = A + B
void matrix_add(int* A_ptr, int* B_ptr, int* C_ptr, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            C_ptr[i * dim + j] = A_ptr[i * dim + j] + B_ptr[i * dim + j];
        }
    }
}
// 减法 C = A - B
void matrix_sub(int* A_ptr, int* B_ptr, int* C_ptr, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            C_ptr[i * dim + j] = A_ptr[i * dim + j] - B_ptr[i * dim + j];
        }
    }
}
// C = A * B
void _matmul_ijk_base(const int* A_ptr, const int* B_ptr, int* C_ptr, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            C_ptr[i * dim + j] = 0; // 初始化
            for (int k = 0; k < dim; ++k) {
                C_ptr[i * dim + j] += A_ptr[i * dim + k] * B_ptr[k * dim + j];
            }
        }
    }
}
void _strassen_matmul_recursive(int* A_start, int a_stride,
                                int* B_start, int b_stride,
                                int* C_start, int c_stride,
                                int current_dim) {
    // 递归终止条件
    if (current_dim <= STRASSEN_THRESHOLD) {

        std::vector<int> tempA(current_dim * current_dim);
        std::vector<int> tempB(current_dim * current_dim);
        std::vector<int> tempC(current_dim * current_dim);

        for(int i = 0; i < current_dim; ++i) {
            for(int j = 0; j < current_dim; ++j) {
                tempA[i * current_dim + j] = A_start[i * a_stride + j];
                tempB[i * current_dim + j] = B_start[i * b_stride + j];
            }
        }
        _matmul_ijk_base(tempA.data(), tempB.data(), tempC.data(), current_dim);
        for(int i = 0; i < current_dim; ++i) {
            for(int j = 0; j < current_dim; ++j) {
                C_start[i * c_stride + j] = tempC[i * current_dim + j];
            }
        }
        return;
    }
    int half_dim = current_dim / 2;
    int* A11 = A_start;
    int* A12 = A_start + half_dim;
    int* A21 = A_start + half_dim * a_stride;
    int* A22 = A_start + half_dim * a_stride + half_dim;
    int* B11 = B_start;
    int* B12 = B_start + half_dim;
    int* B21 = B_start + half_dim * b_stride;
    int* B22 = B_start + half_dim * b_stride + half_dim;
    int* C11 = C_start;
    int* C12 = C_start + half_dim;
    int* C21 = C_start + half_dim * c_stride;
    int* C22 = C_start + half_dim * c_stride + half_dim;
    alignas(16) static int M1[I/2][J/2], M2[I/2][J/2], M3[I/2][J/2], M4[I/2][J/2],
                           M5[I/2][J/2], M6[I/2][J/2], M7[I/2][J/2];
    alignas(16) static int T1[I/2][J/2], T2[I/2][J/2]; // 临时矩阵用于加减
    // 计算 M1 - M7
    // M1 = (A11 + A22) * (B11 + B22)
    matrix_add((int*)A11, (int*)A22, (int*)T1, half_dim); // T1 = A11 + A22
    matrix_add((int*)B11, (int*)B22, (int*)T2, half_dim); // T2 = B11 + B22
    _strassen_matmul_recursive((int*)T1, half_dim, (int*)T2, half_dim, (int*)M1, half_dim, half_dim);
    // M2 = (A21 + A22) * B11
    matrix_add((int*)A21, (int*)A22, (int*)T1, half_dim); // T1 = A21 + A22
    _strassen_matmul_recursive((int*)T1, half_dim, (int*)B11, b_stride, (int*)M2, half_dim, half_dim);
    // M3 = A11 * (B12 - B22)
    matrix_sub((int*)B12, (int*)B22, (int*)T1, half_dim); // T1 = B12 - B22
    _strassen_matmul_recursive((int*)A11, a_stride, (int*)T1, half_dim, (int*)M3, half_dim, half_dim);
    // M4 = A22 * (B21 - B11)
    matrix_sub((int*)B21, (int*)B11, (int*)T1, half_dim); // T1 = B21 - B11
    _strassen_matmul_recursive((int*)A22, a_stride, (int*)T1, half_dim, (int*)M4, half_dim, half_dim);
    // M5 = (A11 + A12) * B22
    matrix_add((int*)A11, (int*)A12, (int*)T1, half_dim); // T1 = A11 + A12
    _strassen_matmul_recursive((int*)T1, half_dim, (int*)B22, b_stride, (int*)M5, half_dim, half_dim);
    // M6 = (A21 - A11) * (B11 + B12)
    matrix_sub((int*)A21, (int*)A11, (int*)T1, half_dim); // T1 = A21 - A11
    matrix_add((int*)B11, (int*)B12, (int*)T2, half_dim); // T2 = B11 + B12
    _strassen_matmul_recursive((int*)T1, half_dim, (int*)T2, half_dim, (int*)M6, half_dim, half_dim);
    // M7 = (A12 - A22) * (B21 + B22)
    matrix_sub((int*)A12, (int*)A22, (int*)T1, half_dim); // T1 = A12 - A22
    matrix_add((int*)B21, (int*)B22, (int*)T2, half_dim); // T2 = B21 + B22
    _strassen_matmul_recursive((int*)T1, half_dim, (int*)T2, half_dim, (int*)M7, half_dim, half_dim);
    // 计算 C11, C12, C21, C22
    // C11 = M1 + M4 - M5 + M7
    matrix_add((int*)M1, (int*)M4, (int*)T1, half_dim); // T1 = M1 + M4
    matrix_sub((int*)T1, (int*)M5, (int*)T2, half_dim); // T2 = T1 - M5
    matrix_add((int*)T2, (int*)M7, (int*)C11, half_dim); // C11 = T2 + M7
    // C12 = M3 + M5
    matrix_add((int*)M3, (int*)M5, (int*)C12, half_dim);
    // C21 = M2 + M4
    matrix_add((int*)M2, (int*)M4, (int*)C21, half_dim);
    // C22 = M1 - M2 + M3 + M6
    matrix_sub((int*)M1, (int*)M2, (int*)T1, half_dim); // T1 = M1 - M2
    matrix_add((int*)T1, (int*)M3, (int*)T2, half_dim); // T2 = T1 + M3
    matrix_add((int*)T2, (int*)M6, (int*)C22, half_dim); // C22 = T2 + M6
}
// Strassen矩阵乘法的公共接口
void matmul_strassen() {
    memset(C, 0, sizeof(C));
    assert(I == K && K == J && (I & (I - 1)) == 0); 
    _strassen_matmul_recursive((int*)A, K, (int*)B, J, (int*)C, J, I);
}
// 多线程分块矩阵乘法
void matmul_tiled_openmp() {
  memset(C, 0, sizeof(C));
  for (int ii = 0; ii < I; ii += BLOCK_SIZE_I) {
    for (int jj = 0; jj < J; jj += BLOCK_SIZE_J) {
      for (int kk = 0; kk < K; kk += BLOCK_SIZE_K) {
        for (int i = ii; i < std::min(ii + BLOCK_SIZE_I, I); i++) {
          for (int j = jj; j < std::min(jj + BLOCK_SIZE_J, J); j++) {
            for (int k = kk; k < std::min(kk + BLOCK_SIZE_K, K); k++) {
              C[i][j] += A[i][k] * B[k][j];
            }
          }
        }
      }
    }
  }
}
void matmul_strassen_openmp() {
    memset(C, 0, sizeof(C));
    assert(I == K && K == J && (I & (I - 1)) == 0); // 检查是否为2的幂且方阵
    int half_dim = I / 2;
    int* A11 = (int*)A;
    int* A12 = (int*)A + half_dim;
    int* A21 = (int*)A + half_dim * K;
    int* A22 = (int*)A + half_dim * K + half_dim;
    int* B11 = (int*)B;
    int* B12 = (int*)B + half_dim;
    int* B21 = (int*)B + half_dim * J;
    int* B22 = (int*)B + half_dim * J + half_dim;
    int* C11 = (int*)C;
    int* C12 = (int*)C + half_dim;
    int* C21 = (int*)C + half_dim * J;
    int* C22 = (int*)C + half_dim * J + half_dim;
    alignas(16) static int M1_par[I/2][J/2], M2_par[I/2][J/2], M3_par[I/2][J/2], M4_par[I/2][J/2],
                           M5_par[I/2][J/2], M6_par[I/2][J/2], M7_par[I/2][J/2];
    alignas(16) static int T1_par[I/2][J/2], T2_par[I/2][J/2];
    // 计算 C11, C12, C21, C22
    // C11 = M1 + M4 - M5 + M7
    matrix_add((int*)M1_par, (int*)M4_par, (int*)T1_par, half_dim);
    matrix_sub((int*)T1_par, (int*)M5_par, (int*)T2_par, half_dim);
    matrix_add((int*)T2_par, (int*)M7_par, (int*)C11, half_dim);
    // C12 = M3 + M5
    matrix_add((int*)M3_par, (int*)M5_par, (int*)C12, half_dim);
    // C21 = M2 + M4
    matrix_add((int*)M2_par, (int*)M4_par, (int*)C21, half_dim);
    // C22 = M1 - M2 + M3 + M6
    matrix_sub((int*)M1_par, (int*)M2_par, (int*)T1_par, half_dim);
    matrix_add((int*)T1_par, (int*)M3_par, (int*)T2_par, half_dim);
    matrix_add((int*)T2_par, (int*)M6_par, (int*)C22, half_dim);
}
int main() {
  init(); 
  constexpr int RUN_TIMES = 3; 
  double total_time_ijk = 0.0f;
  double total_time_ikj = 0.0f;
  double total_time_AT = 0.0f;
  double total_time_BT = 0.0f;
  double total_time_unrolled = 0.0f;
  double total_time_tiled = 0.0f;
  double total_time_write_optimized = 0.0f;
  double total_time_simd = 0.0f;
  double total_time_BT_optimized = 0.0f;
  double total_time_strassen = 0.0f;
  double total_time_tiled_openmp = 0.0f;
  double total_time_strassen_openmp = 0.0f;
  printf("Running %d times for averaging...\n", RUN_TIMES);
  for (int run = 0; run < RUN_TIMES; ++run) {
    auto t = get_time();
    matmul_ijk();
    total_time_ijk += (get_time() - t);
    test();
    t = get_time();
    matmul_ikj();
    total_time_ikj += (get_time() - t);
    test();
    t = get_time();
    matmul_AT();
    total_time_AT += (get_time() - t);
    test();
    t = get_time();
    matmul_BT();
    total_time_BT += (get_time() - t);
    test();
    t = get_time();
    matmul_ikj_unrolled();
    total_time_unrolled += (get_time() - t);
    test();
    t = get_time();
    matmul_tiled();
    total_time_tiled += (get_time() - t);
    test();
    t = get_time();
    matmul_ikj_write_optimized();
    total_time_write_optimized += (get_time() - t);
    test();
    t = get_time();
    matmul_simd();
    total_time_simd += (get_time() - t);
    test();
    t = get_time();
    matmul_BT_optimized();
    total_time_BT_optimized += (get_time() - t);
    test();
    t = get_time();
    matmul_strassen();
    total_time_strassen += (get_time() - t);
    test();
    t = get_time();
    matmul_tiled_openmp();
    total_time_tiled_openmp += (get_time() - t);
    test();
    t = get_time();
    matmul_strassen_openmp();
    total_time_strassen_openmp += (get_time() - t);
    test();
  }
  printf("Average times over %d runs:\n", RUN_TIMES);
  printf("Original ijk matmul time:         %f ms\n", total_time_ijk / RUN_TIMES * 1000);
  printf("Original ikj matmul time:         %f ms\n", total_time_ikj / RUN_TIMES * 1000);
  printf("Original AT matmul time:          %f ms\n", total_time_AT / RUN_TIMES * 1000);
  printf("Original BT matmul time:          %f ms\n", total_time_BT / RUN_TIMES * 1000);
  printf("IKJ Loop Unrolled matmul time:    %f ms\n", total_time_unrolled / RUN_TIMES * 1000);
  printf("Tiled matmul time:                %f ms\n", total_time_tiled / RUN_TIMES * 1000);
  printf("IKJ Write Optimized matmul time:  %f ms\n", total_time_write_optimized / RUN_TIMES * 1000);
  printf("SIMD matmul time (NEON):          %f ms\n", total_time_simd / RUN_TIMES * 1000);
  printf("BT Optimized matmul time (Array Packing): %f ms\n", total_time_BT_optimized / RUN_TIMES * 1000);
  printf("Strassen matmul time:             %f ms\n", total_time_strassen / RUN_TIMES * 1000);
  printf("Tiled OpenMP matmul time:         %f ms\n", total_time_tiled_openmp / RUN_TIMES * 1000);
  printf("Strassen OpenMP matmul time:      %f ms\n", total_time_strassen_openmp / RUN_TIMES * 1000);
  return 0;
}