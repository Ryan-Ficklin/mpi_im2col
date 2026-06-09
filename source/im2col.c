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
MATRIX multiply(MATRIX A, MATRIX B, int N);
void print_matrix(MATRIX M, int N);

//actual program functions
void im2col(MATRIX image, MATRIX col, int H, int W, int K_h, int K_w, int stride);

// main function 
int main(int argc, char **argv)
{
  int rank, nprocs;
  int M_width, M_height, K_width, K_height, stride;
  double start_time, end_time;

  /* counts and displacements for scattering the input matrix */
  int *scatter_counts = NULL;
  int *scatter_displs = NULL;

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

  /* how many rows for each process to distribute  */
  int base = M_height / nprocs;
  int rem = M_height % nprocs;
  int local_rows = base + (rank < rem);
  
  /* column output dimensions, taken from sequential code */
  int out_h = (M_height - K_height) / stride + 1;
  int out_w = (M_width - K_width) / stride + 1;
  /* calculate the total number of elements in our convultion column */
  int col_out_size = out_h * out_w * K_width * K_height;  
  
  /* reading previous input rows according to the height from the kernel */
  int shared_rows = 0;
  /* all except the last rank need to look forward up to their kernel height */
  if (rank < nprocs - 1) {
    shared_rows = K_height - 1;
  }
  int allocated_rows = local_rows + shared_rows; 

  /* get our local M buffer 
   * i.e. the buffer for this process's set of M's rows */
  MATRIX local_M = (MATRIX)scalloc(allocated_rows * M_width,sizeof(float)); 

  /* get our local result buffer 
   * i.e. the buffer for this process to store its col matrix portion */
  //int local_out_h = local_rows; 
  int block_size = out_w * K_width * K_height;
  int local_col_out_size = local_rows * block_size;
  MATRIX result = (MATRIX)scalloc(local_col_out_size,sizeof(float)); 
  
  if(rank == ROOT)
  {
    /* get the counts and displacements for scattering */
    scatter_counts = (int*)smalloc(nprocs * sizeof(int));
    scatter_displs = (int*)smalloc(nprocs * sizeof(int));
    gather_counts = (int*)smalloc(nprocs * sizeof(int));
    gather_displs = (int*)smalloc(nprocs * sizeof(int));

    int scatter_offset = 0;
    int gather_offset = 0;

    /* fill out the count and displacement lists for scattering and gathering */
    for (int i = 0; i < nprocs; i++)
    {
      int rows_i = base + (i < rem);
      /* proportional to local_M */
      scatter_counts[i] = rows_i * M_width;
      scatter_displs[i] = scatter_offset;
      scatter_offset += scatter_counts[i];
      
      /* proportional to result */
      gather_counts[i] = rows_i * block_size;
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
  
  MPI_Scatterv(M, scatter_counts, scatter_displs, MPI_FLOAT, local_M, local_rows * M_width, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  
  /* exchange boundary rows 
   * fetching from rows below us */
  MPI_Request reqs[2];
  int request_count = 0;

  // receiving shared rows 
  if (rank < nprocs - 1 && shared_rows > 0) 
  {
    // recv the rows from the next process 
    MPI_Irecv(local_M + (local_rows * M_width),
              shared_rows * M_width,
              MPI_FLOAT,
              rank + 1,
              0,
              MPI_COMM_WORLD,
              &reqs[request_count++]);
  }

  // sending shared rows 
  if (rank > 0) {
    /* send as many of my rows as the previous process needs */
    /*int rows_to_send = local_rows;
    if (((M_height / nprocs) + (rank - 1 < rem)) >= (K_height - 1)) {
      rows_to_send = K_height - 1;
    }*/
    /* send last row to next process */
    MPI_Isend(local_M, 
              (K_height - 1) * M_width,
              MPI_FLOAT,
              rank - 1,
              0,
              MPI_COMM_WORLD,
              &reqs[request_count++]);
  }

  /* wait for row exchange */
  if (request_count > 0) {
    MPI_Waitall(request_count, reqs, MPI_STATUSES_IGNORE);
  }

  /* im2col calculation */
  int col_idx = 0;

  for (int oh = 0; oh < local_rows; oh++) {
    if ((rank * base + (rank < rem ? rank : rem) + oh) >= out_h) break;

    for (int ow = 0; ow < out_w; ow++) {
      for (int kh = 0; kh < K_height; kh++) {
        for (int kw = 0; kw < K_width; kw++) {
          //col[col_idx++] = image[(oh * stride + kh) * W + (ow * stride + kw)];
          result[col_idx++] = local_M[(oh * stride + kh) * M_width + (ow * stride + kw)];
        }
      }
    }
  }

  MPI_Gatherv(result, local_col_out_size, MPI_FLOAT, col_out, gather_counts, gather_displs, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  
  /* final matrix */
  if(rank == ROOT)
  {
    end_time = MPI_Wtime();
    printf("NxM: %dx%d, Processes: %d, Time: %lf seconds\n", M_width, M_height, nprocs, end_time - start_time);
  }

  /* teardown */
  if (rank == ROOT)
  {
    free(M);
    free(col_out);
    free(scatter_counts);
    free(scatter_displs);
    free(gather_counts);
    free(gather_displs);
  }
  free(local_M);
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
      if(j == W-1) printf("%f", M[i*H + j]);
      else printf("%f, ", M[i*H + j]);
    }

    if(i == H-1) printf("]");
    else printf("],");
  }
  printf("]\n\n");
}
