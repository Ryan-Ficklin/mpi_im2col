#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include "salloc.h"

#define ARGS 2
#define FAIL 1
#define ROOT 0

typedef float * MATRIX;

// helpers for matrix creation/debugging
MATRIX make_matrix(int N, int M);
void print_matrix(MATRIX M, int N);
int check_matrices(MATRIX A, MATRIX B, int size);

//actual program functions
void im2col(MATRIX image, MATRIX col, int H, int W, int K_h, int K_w, int stride);

// main function 
int main(int argc, char **argv)
{
  int rank, nprocs;
  int M_width, M_height, K_width, K_height, stride;
  double start_time, end_time;

  /* counts and displacements for gathering the resultant col */
  int *gather_counts = NULL;
  int *gather_displs = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
  // M: our original matrix
  // col_out: the column which can be used for convolution
  MATRIX M = NULL, col_out = NULL;

  if (argc != 6)
  {
    if (rank == ROOT)
    {
      printf("Usage: mpirun -np P %s <M-Width> <M-Height> <K-Width> <K-Height> <Stride>\n", argv[0]);
    }
    MPI_Finalize();
    return FAIL;
  }
  
  // size of matrix
  M_width = atoi(argv[1]);
  M_height = atoi(argv[2]);
  // size of kernel/filter
  K_width = atoi(argv[3]);
  K_height = atoi(argv[4]);
  stride = atoi(argv[5]);

  /* column output dimensions, taken from sequential code */
  int out_h = (M_height - K_height) / stride + 1; /* number of rows */
  int out_w = (M_width - K_width) / stride + 1;
  /* calculate the total number of elements in our convultion column */
  int col_out_size = out_h * out_w * K_width * K_height;  
  

  /* distribute output rows to each process */
  /* how many rows for each process to distribute  */
  int base = out_h / nprocs;
  int rem = out_h % nprocs;
  int local_rows = base + (rank < rem); 
  int start_offset = 0;
  for (int i = 0; i < rank; i++){
    start_offset += base + (i < rem);
  }


  int local_col_out_size = local_rows * out_w * K_width * K_height;
  MATRIX result = (MATRIX)scalloc(local_col_out_size,sizeof(float)); 
  
  /* allocate space for M broadcast */
  if(rank != ROOT) {
    M = (MATRIX) scalloc(M_width*M_height, sizeof(float));
  }
  if(rank == ROOT)
  {
    /* get the counts and displacements for scattering */
    gather_counts = (int*)smalloc(nprocs * sizeof(int));
    gather_displs = (int*)smalloc(nprocs * sizeof(int));
    int gather_offset = 0;

    /* fill out the count and displacement lists for scattering and gathering */
    for (int i = 0; i < nprocs; i++)
    {
      int rows_i = base + (i < rem);
 
      gather_counts[i] = rows_i * out_w * K_width * K_height;
      gather_displs[i] = gather_offset;
      gather_offset += gather_counts[i];
    }
  
    srand(time(NULL));
    /* get our matrices */
    M = make_matrix(M_width, M_height);
    /* allocate memory for result matrix */
    col_out = (MATRIX) scalloc(col_out_size, sizeof(float));
    /* start timing right before broadcast and scattering starts */
    start_time = MPI_Wtime();
  }

  MPI_Bcast(M, M_width*M_height, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  
  /* im2col calculation */
  int col_idx = 0;

  for (int oh = 0; oh < local_rows; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
      for (int kh = 0; kh < K_height; kh++) {
        for (int kw = 0; kw < K_width; kw++) {
          //col[col_idx++] = image[(oh * stride + kh) * W + (ow * stride + kw)];
          result[col_idx++] = M[((oh + start_offset) * stride + kh) * M_width + (ow * stride + kw)];
        }
      }
    }
  }

  MPI_Gatherv(result, local_col_out_size, MPI_FLOAT, col_out, gather_counts, gather_displs, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  
  /* final matrix */
  if(rank == ROOT)
  {
    end_time = MPI_Wtime();
    printf("NAIVE:\n");
    //printf("Time: %lf seconds\n", M_width, M_height, nprocs, end_time - start_time);
    printf("Time: %lf seconds\n", end_time - start_time);
    /* test if the matrix is the same as sequential */

    MATRIX sequential_col_out = (MATRIX) scalloc(col_out_size, sizeof(float));
    im2col(M, sequential_col_out, M_height, M_width, K_height, K_width, stride);
    
    /* check matrices */
    if (check_matrices(col_out, sequential_col_out, col_out_size) != 0) {
      printf("The matrices differ.\n");
    } else {
      printf("The matrices are the same!\n");
    }
    /* teardown */
    free(sequential_col_out);
  }

  /* teardown */
  if (rank == ROOT)
  {
    free(col_out);
    free(gather_counts);
    free(gather_displs);
  }
  free(M);
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

/* given to matrices and a size, checks if the two are equivalent up to 
 * that size. assumes the matrices are at least of that size, i.e. this 
 * is somewhat unsafe */
int check_matrices(MATRIX A, MATRIX B, int size) {
  int same = 0;
  for (int i = 0; i < size; i++) {
    if (A[i] != B[i]) {
      same = i;
    }
  }
  return same;
}

/* makes an NxM matrix of random float values from 0 - 1 */
MATRIX make_matrix(int N, int M)
{
  int size = N*M;
  MATRIX matrix = (MATRIX) scalloc(size, sizeof(float));

  for (int i = 0; i < size; i++)
  {  
    matrix[i] = (float)rand() / (float)RAND_MAX; 
  }
  return matrix;
}

/* for debugging purposes */
void print_matrix(MATRIX M, int W, int H)
{
  for (int i = 0; i < H; i++){
    
    if(i == 0) printf("[[");
    else printf("\n[");

    for (int j = 0; j < W; j++){
      if(j == W-1) printf("%f", M[i*W + j]);
      else printf("%f, ", M[i*W + j]);
    }

    if(i == H-1) printf("]");
    else printf("],");
  }
  printf("]\n\n");
}
