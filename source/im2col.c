#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include "salloc.h"

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
  int rank, nprocs;
  // NxN matrices 
  int N;
  double start_time, end_time;
  int *counts = NULL;
  int *displs = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MATRIX A, B, M;

  if (argc != 2)
  {
    if (rank == ROOT)
    {
      printf("Usage: mpirun -np P %s N\n", argv[0]);
    }
    MPI_Finalize();
    return FAIL;
  }
  
  N = atoi(argv[1]);

  /* how many rows for each process */
  int base = N / nprocs;
  int rem = N % nprocs;
  int local_rows = base + (rank < rem);
  int local_size = local_rows * N;
  /* get our local A buffer */
  MATRIX local_A = (MATRIX)scalloc(local_size,sizeof(float)); 
  /* get our local buffer for result */
  MATRIX result = (MATRIX)scalloc(local_size,sizeof(float));
  
  /* every rank needs space for B as well. ROOT will be the only process
   * to actually make the matrix originally */
  if(rank != ROOT)
  {
    B = (MATRIX) scalloc(N*N,sizeof(float));
  }

  if(rank == ROOT)
  {
    /* get the counts and displacements for scattering */
    counts = (int*)smalloc(nprocs * sizeof(int));
    displs = (int*)smalloc(nprocs * sizeof(int));

    int off = 0;
    for (int i = 0; i < nprocs; i++)
    {
      int rows_i = base + (i < rem);
      counts[i] = rows_i * N;
      displs[i] = off;
      off += counts[i];
    }
  
    srand(time(NULL));
    /* get our matrices */
    A = make_matrix(N);
    B = make_matrix(N);
    /* allocate memory for result matrix */
    M = (MATRIX) scalloc(N*N, sizeof(float));
    /* start timing right before broadcast and scattering starts */
    start_time = MPI_Wtime();
  }
  
  /* just give B to everyone else. Block decomp would be better, oh well */
  MPI_Bcast(B, N*N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  MPI_Scatterv(A, counts, displs, MPI_FLOAT, local_A, local_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

  /* same matrix computation as before */
  for(int r = 0; r < local_rows; r++){
    for(int c = 0; c < N; c++){
      for(int i = 0; i < N; i++){
        result[r*N + c] += local_A[r*N + i] * B[i*N + c];
      }
    }
  }

  MPI_Gatherv(result, local_size, MPI_FLOAT, M, counts, displs, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  
  /* final matrix */
  if(rank == ROOT)
  {
    end_time = MPI_Wtime();
    printf("N: %d, Processes: %d, Time: %lf seconds\n", N, nprocs, end_time - start_time);
    //print_matrix(A, N);
    //print_matrix(B, N);
    //print_matrix(M, N);
  }

  /* teardown */
  if (rank == ROOT)
  {
    free(A);
    free(M);
    free(counts);
    free(displs);
  }
  free(B);
  free(local_A);
  free(result);

  MPI_Finalize();
  
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

  for (int oh = 0; oh < out_h; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
      for (int kh = 0; kh < K_h; kh++) {
        for (int kw = 0; kw < K_w; kw++) {
          col[col_idx++] = image[(oh * stride + kh) * W + (ow * stride + kw)];
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
    matrix[i] = (float)rand() / (float)RAND_MAX; 
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
      if(j == N-1) printf("%f", M[i*N + j]);
      else printf("%f, ", M[i*N + j]);
    }

    if(i == N-1) printf("]");
    else printf("],");
  }
  printf("]\n\n");
}
