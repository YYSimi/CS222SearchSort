#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common/io.h"
#include <time.h>
#include <cuda.h>
#include "common/cuPrintf.cu"

/*********************** Data Definitions ********************************/
#define THREADS_PER_BLOCK 64

//These Inline Functions are used in the CPU Quicksort Implementation
#define swap(A,B) { float temp = A; A = B; B = temp;}
//#define compswap(A,B) if(B < A) swap(A,B)

//These Data Structs are used in the GPU Quicksort Implementation
typedef enum compstatus{
  LEQ = 0,
  GT,
  UNSORTED
} cstate;

typedef struct data{
  float val;
  cstate status;
} data;

typedef struct vars{
  int l;
  int r;
  int leq;
} vars;

/*********************** CPU QUICKSORT IMPLEMENTATION ***********************/

/* csort
 *
 * This function is an implementation of 'Quicksort with three-way 
 * partitioning' from 'Algorithms in C' (Program 7.5, page 326).
 *
 * Parameters:
 * ls: The list of floating points being sorted
 * l: index of the left most item in ls being sorted at the moment
 * r: index of the right most item in ls being sorted at the moment
 */
void csort(float ls[], int l, int r){
  int i, j, k, p, q;
  float v;
  if(r <= l)
    return;
  v = ls[r];
  i = l-1;
  j = r;
  p = l-1;
  q = r;
  for(;;){
    while(ls[++i]< v);
    while(v < ls[--j])
      if(j == 1)
	break;
    if(i >= j)
      break;
    swap(ls[i],ls[j]);
    if(ls[i] == v){
      p++;
      swap(ls[p], ls[i]);
    }
    if(v == ls[j]){
      q--;
      swap(ls[q],ls[j]);
    }
  }
  swap(ls[i],ls[r]);
  j = i-1;
  i++;
  for(k = l; k < p; k++,j--)
    swap(ls[k],ls[j]);
  for(k = r-1; k > q; k--,i++)
    swap(ls[k], ls[i]);

  csort(ls, l, j);
  csort(ls, i, r);
}

/* cpu_quicksort
 *
 * This function is called to sort the floating point array using a CPU-based
 * implementation of quicksort. Its purpose is to set up the timing functions
 * to wrap the recursive 'csort' function which does the actual sorting
 *
 * Parameters:
 * unsorted: The array of floating point numbers to be sorted
 * length: the length of the unsorted & sorted arrays
 * sorted: an output parameter, will store the final, sorted array.
 *
 * Output:
 * time: This function should return the amount of time taken to sort the list.
 */
double cpu_quicksort(float unsorted[], int length, float sorted[]){

  for(int i = 0; i < length; i++)
    sorted[i] = unsorted[i];

  clock_t start, end;
  double time;
  start = clock();
  csort(sorted, 0, length - 1);
  end = clock();
  time = ((double) end - start) / CLOCKS_PER_SEC;

  return time;
}

/***************************** GPU IMPLEMENTATION ****************************/

/* gpuPartitionSwap
 *
 * This kernel function is called recursively by the host. Its purpose is to, 
 * given a pivot value, partition and swap items in the section of the input
 * array bounded by the l & r indices, then store the pivot in the correct
 * location.
 *
 * Parameters:
 * input: The unsorted (or partially sorted) input data
 * output: The aptly named output parameter, it is the same as input, but all
 *         floating points within (l,r) have been partitioned and swapped.
 * endpts: This is a custom data struct meant to 
 *         a) hold a counter variable in global memory
 *         b) pass the l' and r' parameters back to the host to the left and
 *            right of the positioned pivot item.
 * pivot: This is the pivot value, about which all items in (l,r) are being
 *        swapped.
 * l: the left index bound on input & output
 * r: the right index bound on input & output
 * d_leq: an array of offset values, storedin global device memory
 * nBlocks: The total number of blocks, to be used to determine the location
 *          of insertion of the pivot.
 *
 */
__global__ void gpuPartitionSwap(data * input, data * output, vars * endpts, 
				 float pivot, int l, int r, int d_leq[], 
				 int nBlocks)
{
  //copy a section of the input into shared memory
  __shared__ data bInput[THREADS_PER_BLOCK];
  __shared__ int leq;
  leq = 0;
  int idx = l + blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
  if(idx <= r){
    bInput[threadIdx.x] = input[l+ blockIdx.x*THREADS_PER_BLOCK + threadIdx.x];

    //make comparison against the pivot, setting 'status' and updating the counter (if necessary)
    if( (bInput[threadIdx.x]).val <= pivot ){
      (bInput[threadIdx.x]).status = LEQ;
      (endpts->leq)++;
      leq++;
    } else
      (bInput[threadIdx.x]).status = GT;
    
  }
  __syncthreads();
  if((idx <= r) && (threadIdx.x == 0)){     
    //write the update
      d_leq[blockIdx.x] = leq;
  }      
  __syncthreads();
  if((idx <= r) && (threadIdx.x == 0)){
    int lOffset = 0;
    int rOffset = 0;
    for(int i = 0; i <= blockIdx.x; i++){
      lOffset += d_leq[i];
      rOffset += (THREADS_PER_BLOCK - (d_leq[i]+1));
    }
    
    int m = 0;
    int n = 0;
    for(int j = 0; j <= THREADS_PER_BLOCK; j++){
      if(bInput[j].status == LEQ){
	output[lOffset+m] = bInput[j];
	m++;
      } else {
	output[rOffset - n] = bInput[j];
	n++;
      }
    }
  }

  __syncthreads();
  cuPrintf("idx: %d, threadIdx.x: %d, blockIdx.x: %d\n", idx, threadIdx.x, blockIdx.x);
  if((idx <= r) && (threadIdx.x == 0) && (blockIdx.x == 0)){
    int pOffset = l;
    for(int k = 0; k < nBlocks; k++)
      pOffset += d_leq[k];

    cuPrintf("%d", pOffset);

    data p;
    p.val = pivot;
    output[pOffset] = p;
    endpts->l = (pOffset - 1);
    endpts->r = (pOffset + 1);
  }

  return;
}

void gqSort(data * ls, int l, int r, int length){
  //if (r - l) > 1
  if((r - l) > 1){
    //1. grab pivot
    float pivot = ls[r].val;
    printf("(l, r): (%d, %d)\n", l, r);

    //2. set-up gpu vars
    int numBlocks = (r - l) / THREADS_PER_BLOCK;
    if((numBlocks * THREADS_PER_BLOCK) < (length - 1))
      numBlocks++;

    printf("numBlocks:%d\n", numBlocks);
    data * d_ls;
    data * d_ls2;
    vars endpts;

    vars * d_endpts;
    int * d_leq;
    cudaMalloc(&(d_ls), sizeof(data)*length);
    cudaMalloc(&(d_ls2), sizeof(data)*length);
    cudaMalloc(&(d_endpts), sizeof(vars));
    cudaMalloc(&(d_leq), 4*numBlocks);
    cudaMemcpy(d_ls, ls, sizeof(data)*length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls2, ls, sizeof(data)*length, cudaMemcpyHostToDevice);

    //3. call gpuPartition function
    gpuPartitionSwap<<<numBlocks, THREADS_PER_BLOCK>>>(d_ls, d_ls2, d_endpts, pivot, l, r, d_leq, numBlocks);

    //4. Retrieve sorted list and other variables
    cudaMemcpy(ls, d_ls2, sizeof(data)*length, cudaMemcpyDeviceToHost);
    cudaMemcpy(&(endpts), d_endpts, sizeof(vars), cudaMemcpyDeviceToHost);
    printf("new endpoints: l' = %d, r' = %d\n", endpts.l, endpts.r);

    //5.recursively call on left/right sections of list generated by gpuPartition
    gqSort(ls, l, endpts.l, length);
    gqSort(ls, endpts.r, r, length);
  }
  return;
}

/* gpu_quicksort
 *
 * This is a function meant to set up the custom 'data' struct array
 * used by the gpu implementation of quicksort, as well as to calculate
 * the time of execution of the sorting algorithm.
 *
 * Parameters:
 * unsorted: The array of floats to be sorted
 * length: The length of the unsorted and sorted arrays
 * sorted: An output parameter, to be filled with the sorted array.
 *
 * Output:
 * time: This function returns the time of execution required by the
 *       sorting algorithm
 */
double gpu_quicksort(float unsorted[], int length, float sorted[]){
  time_t start, end;
  double time;    

  data list[length];
  for(int i = 0; i < length; i++){
    list[i].val = unsorted[i];
    list[i].status = UNSORTED;
  }
  start = clock();
  gqSort(list, 0, length - 1, length);
  end = clock();
  time = ((double) end - start) / CLOCKS_PER_SEC;

  for(int j = 0; j < length; j++)
    sorted[j] = list[j].val;

  return time;
}

/* quicksort
 * 
 * This function is called by main to populate a result, testing the CPU
 * and GPU implementations of quicksort.
 *
 * Parameters:
 * unsorted: an unsorted array of floating points
 * length: the length of the unsorted array
 * result: an output parameter to be filled with the results of the cpu and gpu
 *         implementations of quicksort.
 *
 */
void quicksort(float unsorted[], int length, Result * result){
  result = (Result *) malloc(sizeof(Result));

  cudaPrintfInit();
  
  if(result == NULL){
    fprintf(stderr, "Out of Memory\n");
    exit(1);
  }
  strcpy(result->tname, "Quick Sort");
  float sorted[2][length];

  result->cpu_time = cpu_quicksort(unsorted, length, sorted[0]);
  result->gpu_time = gpu_quicksort(unsorted, length, sorted[1]);

  //check that sorted[0] = sorted[1];
  int n = 0;
  for(int i = 0; i < length; i++){
    if(sorted[0][i] != sorted[1][i])
      n++;
    //    printf("CPU #%d: %f\t", i, sorted[0][i]); 
    //    printf("GPU #%d: %f", i, sorted[1][i]); 
    //    printf("\n", i, sorted[0][i]); 
  }

  cudaThreadSynchronize();
  cudaPrintfDisplay(stdout,true);
  cudaPrintfEnd();

  if(n!= 0){
    fprintf(stdout, "There were %d discrepencies between the CPU and GPU QuickSort algorithms\n", n);
  }

  return;
}
