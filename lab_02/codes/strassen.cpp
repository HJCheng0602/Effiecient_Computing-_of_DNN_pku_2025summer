#include <vector>
#include <cassert>
#include <cstring>
#include <random>
#include <sys/time.h>
#include <iostream> 
const int N = 256; // 统一矩阵维度为N，因为Strassen算法要求方阵
const int STRASSEN_THRESHOLD = 64; 
std::vector<int> A(N * N);
std::vector<int> B(N * N);
std::vector<int> C(N * N);
std::vector<int> C_groundtruth(N * N);
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}
void init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-5, 5); 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = distrib(gen);
            B[i * N + j] = distrib(gen);
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            long long sum = 0;
            for (int k = 0; k < N; k++) {
                sum += (long long)A[i * N + k] * B[k * N + j];
            }
            C_groundtruth[i * N + j] = static_cast<int>(sum);
        }
    }
}
void test() {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      assert(C[i * N + j] == C_groundtruth[i * N + j]);
    }
  }
}
void matrix_add_strided(const int* A_ptr, int a_stride,
                        const int* B_ptr, int b_stride,
                        int* C_ptr, int c_stride,
                        int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            C_ptr[i * c_stride + j] = A_ptr[i * a_stride + j] + B_ptr[i * b_stride + j];
        }
    }
}
void matrix_sub_strided(const int* A_ptr, int a_stride,
                        const int* B_ptr, int b_stride,
                        int* C_ptr, int c_stride,
                        int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            C_ptr[i * c_stride + j] = A_ptr[i * a_stride + j] - B_ptr[i * b_stride + j];
        }
    }
}
void matmul_base_strided(const int* A_ptr, int a_stride,
                         const int* B_ptr, int b_stride,
                         int* C_ptr, int c_stride,
                         int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            long long sum = 0;
            for (int k = 0; k < dim; ++k) {
                sum += (long long)A_ptr[i * a_stride + k] * B_ptr[k * b_stride + j];
            }
            C_ptr[i * c_stride + j] = static_cast<int>(sum);
        }
    }
}
void _strassen_matmul_recursive(const int* A_start, int a_stride,
                                const int* B_start, int b_stride,
                                int* C_start, int c_stride,
                                int current_dim) {
    if (current_dim <= STRASSEN_THRESHOLD) {
        matmul_base_strided(A_start, a_stride, B_start, b_stride, C_start, c_stride, current_dim);
        return;
    }
    int half_dim = current_dim / 2;
    const int* A11 = A_start;
    const int* A12 = A_start + half_dim;
    const int* A21 = A_start + half_dim * a_stride;
    const int* A22 = A_start + half_dim * a_stride + half_dim;
    const int* B11 = B_start;
    const int* B12 = B_start + half_dim;
    const int* B21 = B_start + half_dim * b_stride;
    const int* B22 = B_start + half_dim * b_stride + half_dim;
    int* C11 = C_start;
    int* C12 = C_start + half_dim;
    int* C21 = C_start + half_dim * c_stride;
    int* C22 = C_start + half_dim * c_stride + half_dim;
    std::vector<int> M1(half_dim * half_dim);
    std::vector<int> M2(half_dim * half_dim);
    std::vector<int> M3(half_dim * half_dim);
    std::vector<int> M4(half_dim * half_dim);
    std::vector<int> M5(half_dim * half_dim);
    std::vector<int> M6(half_dim * half_dim);
    std::vector<int> M7(half_dim * half_dim);
    std::vector<int> T1(half_dim * half_dim);
    std::vector<int> T2(half_dim * half_dim);
    matrix_add_strided(A11, a_stride, A22, a_stride, T1.data(), half_dim, half_dim);
    matrix_add_strided(B11, b_stride, B22, b_stride, T2.data(), half_dim, half_dim);
    _strassen_matmul_recursive(T1.data(), half_dim, T2.data(), half_dim, M1.data(), half_dim, half_dim);
    matrix_add_strided(A21, a_stride, A22, a_stride, T1.data(), half_dim, half_dim);
    _strassen_matmul_recursive(T1.data(), half_dim, B11, b_stride, M2.data(), half_dim, half_dim);
    matrix_sub_strided(B12, b_stride, B22, b_stride, T1.data(), half_dim, half_dim);
    _strassen_matmul_recursive(A11, a_stride, T1.data(), half_dim, M3.data(), half_dim, half_dim);
    matrix_sub_strided(B21, b_stride, B11, b_stride, T1.data(), half_dim, half_dim);
    _strassen_matmul_recursive(A22, a_stride, T1.data(), half_dim, M4.data(), half_dim, half_dim);
    matrix_add_strided(A11, a_stride, A12, a_stride, T1.data(), half_dim, half_dim);
    _strassen_matmul_recursive(T1.data(), half_dim, B22, b_stride, M5.data(), half_dim, half_dim);
    matrix_sub_strided(A21, a_stride, A11, a_stride, T1.data(), half_dim, half_dim);
    matrix_add_strided(B11, b_stride, B12, b_stride, T2.data(), half_dim, half_dim);
    _strassen_matmul_recursive(T1.data(), half_dim, T2.data(), half_dim, M6.data(), half_dim, half_dim);
    matrix_sub_strided(A12, a_stride, A22, a_stride, T1.data(), half_dim, half_dim);
    matrix_add_strided(B21, b_stride, B22, b_stride, T2.data(), half_dim, half_dim);
    _strassen_matmul_recursive(T1.data(), half_dim, T2.data(), half_dim, M7.data(), half_dim, half_dim);
    matrix_add_strided(M1.data(), half_dim, M4.data(), half_dim, T1.data(), half_dim, half_dim);
    matrix_sub_strided(T1.data(), half_dim, M5.data(), half_dim, T2.data(), half_dim, half_dim);
    matrix_add_strided(T2.data(), half_dim, M7.data(), half_dim, C11, c_stride, half_dim);
    matrix_add_strided(M3.data(), half_dim, M5.data(), half_dim, C12, c_stride, half_dim);
    matrix_add_strided(M2.data(), half_dim, M4.data(), half_dim, C21, c_stride, half_dim);
    matrix_sub_strided(M1.data(), half_dim, M2.data(), half_dim, T1.data(), half_dim, half_dim);
    matrix_add_strided(T1.data(), half_dim, M3.data(), half_dim, T2.data(), half_dim, half_dim);
    matrix_add_strided(T2.data(), half_dim, M6.data(), half_dim, C22, c_stride, half_dim);
}
void matmul_strassen() {
    assert(N > 0 && (N & (N - 1)) == 0); 
    _strassen_matmul_recursive(A.data(), N, B.data(), N, C.data(), N, N);
}
void matmul_ikj() {
    std::fill(C.begin(), C.end(), 0);
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
    }
  }

void matmul_ijk() {
    std::fill(C.begin(), C.end(), 0);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i* N + j] += A[i * N + k] * B[k * N + j];    
      }   
    }
  }
}
void matmul_AT() {
    // For this code, we assume square matrices of size N x N
    // AT is the transpose of A
    std::vector<int> AT(N * N, 0);
    std::fill(C.begin(), C.end(), 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            AT[i * N + j] = A[j * N + i];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += AT[k * N + i] * B[k * N + j];
            }
        }
    }
}
void matmul_BT() {
    // For this code, we assume square matrices of size N x N
    // BT is the transpose of B
    std::vector<int> BT(N * N, 0);
    std::fill(C.begin(), C.end(), 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BT[i * N + j] = B[j * N + i];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * BT[j * N + k];
            }
        }
    }
}
int main()
{
    int RUN_TIMES = 10;
    init();
    double total_time_ikj = 0.0f;
    double total_time_strassen = 0.0f;
    double total_time_ijk = 0.0f;
    double total_time_AT = 0.0f;
    double total_time_BT = 0.0f;
    std::cout.precision(3);

    std::cout << "Running " << RUN_TIMES << " times for averaging..." << std::endl;
    for(int run = 0; run < RUN_TIMES; ++run) {
        auto t = get_time();
        matmul_ikj();
        total_time_ikj += (get_time() - t);
        test();

        t = get_time();
        matmul_ijk();
        total_time_ijk += (get_time() - t);
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
        matmul_strassen();
        total_time_strassen += (get_time() - t);
        test();
    }
    std::cout.precision(3);
    std::cout << std::fixed;
    std::cout << "Average time for matmul_ikj: " << total_time_ikj / RUN_TIMES << " seconds" << std::endl;
    std::cout << "Average time for matmul_ijk: " << total_time_ijk / RUN_TIMES << " seconds" << std::endl;
    std::cout << "Average time for matmul_AT: " << total_time_AT / RUN_TIMES << " seconds" << std::endl;
    std::cout << "Average time for matmul_BT: " << total_time_BT / RUN_TIMES << " seconds" << std::endl;
    std::cout << "Average time for matmul_strassen: " << total_time_strassen / RUN_TIMES
                << " seconds" << std::endl;
    std::cout << "All tests passed!" << std::endl;

    return 0;
}
