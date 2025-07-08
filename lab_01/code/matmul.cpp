// g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall && ./matmul

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int I = 256;
constexpr int K = 256;
constexpr int J = 256;
int A[I][K];
int B[K][J];
int BT[J][K];
int AT[K][I];
int C[I][J];
int C_groundtruth[I][J];

void init() {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < K; j++) {
      A[i][j] = rand(); 
      
    } 
  }
  for (int i = 0; i < K; i++)
  {
    for(int j = 0; j < J; j++)
    {
      B[i][j] = rand();
    }
  }
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        C_groundtruth[i][j] += A[i][k] * B[k][j];
      }
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

void matmul() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

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

int main() {
  init();
  float avg_time1 = 0.0f;
  float avg_time2 = 0.0f;
  float avg_time = 0.0f;
  float avg_time3 = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t1 = get_time();
     //matmul_ikj();
    matmul(); 
    //matmul_AT();
    //matmul_BT();
    test();
    //printf("%f\n", get_time() - t);
    avg_time1 += get_time() - t1;

    auto t2 = get_time();
    matmul_ikj();
    test();
    avg_time2 += get_time() - t2;

    auto t3 = get_time();
    matmul_AT();
    test();
    avg_time3 += get_time() - t3;

    auto t4 = get_time();
    matmul_BT();
    test();
    avg_time += get_time() - t4;
  }
  printf("symple matmul time: %f ms\n", avg_time1 / 32 * 1000);
  printf("ikj matmul time: %f ms\n", avg_time2 / 32 * 1000);
  printf("AT matmul time: %f ms\n", avg_time3 / 32 * 1000);
  printf("BT matmul time: %f ms\n", avg_time / 32 * 1000);
  return 0;
}

