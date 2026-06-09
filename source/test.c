#include <stdio.h>
//#include <mpi.h>
#include <time.h>
#include "salloc.h"
#include <math.h>

#define ARGS 2
#define FAIL 1
#define ROOT 0

typedef float * MATRIX;

// helpers for matrix creation/debugging
MATRIX make_matrix(int N);
MATRIX multiply(MATRIX A, MATRIX B, int N);
void print_matrix(MATRIX M, int N);

//actual program functions
void im2col(MATRIX image, MATRIX col, int H, int W, int K_h, int K_w, int stride);

// main function 
int main(int argc, char **argv)
{
  int i = 0;
  /*float A[] = {0, 1, 2, 3, 4, 5,
               6, 7, 8, 9, 10, 11,
               12, 13, 14, 15, 16, 17, 
               18, 19, 20, 21, 22, 23, 
               24, 25, 26, 27, 28, 29, 
               30, 31, 32, 33, 34, 35};
  */
  // srand(0);
  MATRIX A = make_matrix(10);
  MATRIX col = (MATRIX) scalloc(100, 100);
  int N = 6;
  int k = 2;
  int s = 2;
  int c = ((N - k) / s + 1) * ((N - k) / s + 1) * k * k;
  int csqrt = sqrt(c);
  printf("c: %d\n", c);

  im2col(A, col, N, N, k, k, s);
  printf("\n");
  print_matrix(col, csqrt);
  free(A);
  free(col);
  return 0;
}

/* naive (sequential) im2col 
 * provided in the assignment specifications */
void im2col(
  MATRIX image, 
  MATRIX col,
  int H, //image height

  int W,//image width
  int K_h, //kernel height

  int K_w, //kernel width
  int stride)
{
  int out_h = (H - K_h) / stride + 1;
  int out_w = (W - K_w) / stride + 1;

  int col_idx = 0;
  int img_idx;

  for (int oh = 0; oh < out_h; oh++) {
    //printf("\n%d \n", oh);
    for (int ow = 0; ow < out_w; ow++) {
      //printf("%d ", ow);
      for (int kh = 0; kh < K_h; kh++) {
        for (int kw = 0; kw < K_w; kw++) {
          img_idx = (oh * stride + kh) * W + (ow * stride + kw);
          printf("col_idx: %d, img_idx: %d\n", col_idx, img_idx);
          col[col_idx++] = image[img_idx];
        }
      }
    }
  }
}

/* makes an NxN matrix of random float values from 0 - 1 */
MATRIX make_matrix(int N)
{
  int size = N*N;
  MATRIX matrix = (MATRIX) scalloc(size, sizeof(float));

  for (int i = 0; i < size; i++)
  {  
    //matrix[i] = (float)rand() / (float)RAND_MAX; 
    matrix[i] = i;
  }
  return matrix;
}

/* for debugging purposes */
void print_matrix(MATRIX M, int N)
{
  for (int i = 0; i < N; i++){
    
    if(i == 0) printf("[[");
    else printf("\n[");

    for (int j = 0; j < N; j++){
      if(j == N-1) printf("%.0f", M[i*N + j]);
      else printf("%.0f, ", M[i*N + j]);
    }

    if(i == N-1) printf("]");
    else printf("],");
  }
  printf("]\n\n");
}
