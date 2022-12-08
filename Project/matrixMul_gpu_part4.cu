/*
 * _MATRIXMUL_GPU_CU_
 *
 * 2022 Mert SIDE
 *
 * CS5375 Computer Systems Organization and Architecture 
 * Guest Lecture: GPU Programming
 *
 * Multiplying two matrices on the GPU
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// ------------------------------------------------------------------ GPUmatmul
__global__
void GPUmatmul(int N, double *x, double *y, double *ans)
{

int column = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

ans[row*N + column] = 0;
      for (int k=0;k<N;k++)
        {
            ans[row*N + column] += x[row*N + k] * y[k*N + column];
        }
}

// ---------------------------------------------------------------------- check
bool check(int N, double *ans)
{
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      if(ans[i*N+j] != 20.0) return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------- MAIN
int main(void)
{
  // size of matrix
  int N = 1<<9; // binary left-shift: 1 * 2^9 = 512
  printf("Size of matrix (N) is %d by %d.\n", N, N);
  int iter = 3;
  clock_t t;
  
  // Martices
  double *x, *y, *ans;

  // TODO: Allocate Unified Memory - accessible from both CPU and GPU
  cudaMallocManaged((void **)&x, sizeof(float) * N * N);
  cudaMallocManaged((void **)&y, sizeof(float) * N * N);
  cudaMallocManaged((void **)&ans, sizeof(float) * N * N);

  int device = -1;

  cudaMemPrefetchAsync(x, sizeof(float) * N * N, device, NULL);
  cudaMemPrefetchAsync(y, sizeof(float) * N * N, device, NULL);
  cudaMemPrefetchAsync(ans, sizeof(float) * N * N, device, NULL);
  // ..........................................................................
  // initialize x,y and ans arrays on the host
  for (int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      x[i*N+j] = 5;
      y[i*N+j] = (i==j?1:0);
      ans[i*N+j] = (double)0.000000000000;
    }
  }

  int BLOCK_SIZE = 16; // 16*16 = 256
  int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

  dim3 grid(GRID_SIZE, GRID_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  // ..........................................................................
  double avg=0;
  std::cout<<"Starting unoptimized GPU computation"<<std::endl;
  // Run kernel on GPU
  for(int i = 0; i <= iter; i++) {
    t = clock();

    GPUmatmul<<<grid,threads>>>(N, x, y,ans);

    cudaDeviceSynchronize();
    t = clock() - t;
    if(i) avg += t; //we will ignore the first run
    // printf ("It took GPU-%d %f ms.\n",i,(((double)t)/CLOCKS_PER_SEC)*1000);
  }
  
  avg = t;
  avg /= iter;
  avg /= CLOCKS_PER_SEC;
  avg *= 1000;
  printf("It took %lf ms on avg.\n", avg);
  if(check(N,ans)) std::cout<<"RUN OK."<<std::endl;
  else std::cout<<"RUN NOT OK."<<std::endl;

  // ..........................................................................
  
  // TODO: Free memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(ans);

  return 0;
}
/* EOF */