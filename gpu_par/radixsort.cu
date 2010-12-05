#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"
#include <time.h>
#include <cuda.h>


#define THREADS_PER_BLOCK 64
#define M 10

//CPU Radix Sort
#define swap(A,B) { float temp = A; A = B; B = temp;}
#define compswap(A,B) if(B < A) swap(A,B)
#define digit(A,B) (((A) >> (8 - ((B)+1) * 8)) & ((1 << 8) - 1))
#define ch(A) digit(A, D)

//to perform radix sort on our lists of positve single-precision floating points,
//we want to flip the sign bit and treat them as unsigned integers
void Flip(float list[], uint flipped[], int length){
  for(int i = 0; i < length; i++){
    uint temp = (uint) list[i];
    uint mask = -((int) ( temp >> 31)) | 0x80000000;
    flipped[i] = mask ^ ((uint) temp);
  }
  return;
}

// Having now sorted the uint's, we want to flip the sign bit back
// and treat them as floats once more.
void unFlip(uint list[], float unFlipped[], int length){
  for(int i = 0; i< length; i++){
    uint temp = list[i];
    uint mask = ((temp >> 31) - 1) | 0x80000000;
    unFlipped[i] = (float) (mask ^ temp);
  }
  return;
}

void insert(uint ls[], int l, int r){
  int i;
  for(i = r; i > l; i--) 
    compswap(ls[i-1], ls[i]);
  for(i = l + 2; i <= r; i++){
    int j = i;
    uint v = ls[i];
    while(v < ls[j-1]){
      ls[j] = ls[j-1];
      j--;
    }
    ls[j] = v;
  }
  return;
}

void RadixQuicksort(uint ls[], int l, int r, int D){
  int i, j, k, p, q, v;
  if(r-1 <= M){
    insert(ls, l, r);
    return;
  }
  v = ch(ls[r]);
  i = l-1;
  j = r;
  p = l-1;
  q = r;
  while(i < j){
    while(ch(ls[++i]) < v);
    while (v < ch(ls[--j]))
      if(j == 1)
	break;
    if(i > j)
      break;
    swap(ls[i],ls[j]);
    if(ch(ls[i]) == v){
      p++;
      swap(ls[p],ls[i]);
    }
    if(ch(ls[j]) == v){
      q--;
      swap(ls[j], ls[q]);
    }
  }
  if(p == q){
    if(v != '\0'){
      RadixQuicksort(ls, l, r, D+1);
      return;
    }
    if(ch(ls[i]) < v)
      i++;
    for(k=1; k <=p; k++, j--)
      swap(ls[k], ls[j]);
    for(k=r; k >= q; k--, i++)
      swap(ls[k], ls[i]);
    RadixQuicksort(ls, l, j, D);
    if((i == r) && (ch(ls[i]) == v))
      i++;
    if( v != '\0')
      RadixQuicksort(ls, j+1, i-1, D+1);
    RadixQuicksort(ls, i, r, D);
  }
  return;
}

double cpu_radixsort(float * unsorted, int length, float * sorted){
  //1. Convert float * unsorted to uint *
  uint flipped[length];
  Flip(unsorted, flipped, length);

  //2. Perform Radix Sort
  time_t start, end;
  double time;
  start = clock();

  //radix_sort call
  RadixQuicksort(flipped, 0, length - 1, 0);

  end = clock();
  time = ((double) end - start) / CLOCKS_PER_SEC;

  //3. Convert uint * to float *
  unFlip(flipped, sorted, length);

  return time;
}


//GPU Radix Sort
__device__ uint devFlip(uint u, int D){
  if(!D){
    uint mask = 0x80000000;
    return mask ^ u;
  }
  return u;
}

__device__ float devUnFlip(uint u, int D){
  if(D == 31){
     uint mask = ((u >>31) - 1) | 0x80000000;
     return (float) (mask ^ u);
  }
  return u;
}
typedef enum {
  BUCKET0 = 0,
  BUCKET1
}state;

typedef struct data{
  uint val;
  state bucket;
} data;

__global__ void gpuRadixBitSort(data input[], data output[], int l, int r, int nZeroes[], int D){
  __shared__ data bInput[THREADS_PER_BLOCK];
  __shared__ int zeroes;
  zeroes = 0;
  int idx = l+blockIdx.x*THREADS_PER_BLOCK+threadIdx.x;
  if(idx <= r){
    bInput[threadIdx.x] = input[idx];
    devFlip(bInput[threadIdx.x].val, D);
    uint f = bInput[threadIdx.x].val;
    f = f << (0 + D);
    f = f >> (31 - D);
    if(f == 0){
      bInput[threadIdx.x].bucket = BUCKET0;
      zeroes++;
    } else
      bInput[threadIdx.x].bucket = BUCKET1;

    __syncthreads();

    if(threadIdx.x == 0){
      nZeroes[blockIdx.x] = zeroes;
      int lOffset = l;
      int rOffset = r;
      int i;
      for(i = 0; i <= blockIdx.x; i++){
	lOffset += nZeroes[i];
	rOffset -= THREADS_PER_BLOCK - nZeroes[i];
      }
      int m = 0;
      int n = 0;
      int j;
      for(j = 0; j <= THREADS_PER_BLOCK; j++){
	if(bInput[j].bucket == BUCKET0){
	  output[lOffset+m] = bInput[j];
	  m++;
	} else {
	  output[rOffset - n] = bInput[j];
	  n++;
	}
      }
      __syncthreads();
    }
    devUnFlip(output[idx].val, D);
  }
  return;
}

void grSort(data list[], int l, int r, int length, int D){
  if(D < 31){
    int numBlocks = (r - l + 1) / THREADS_PER_BLOCK;

    data * d_list;
    data * d_list2;
    int * d_divide;
    int divide[numBlocks];
    cudaMalloc(&d_list, sizeof(data)*length);
    cudaMalloc(&d_list2, sizeof(data)*length);
    cudaMalloc(&d_divide, sizeof(int)*numBlocks);
    cudaMemcpy(d_list, list, sizeof(data)*length, cudaMemcpyHostToDevice);
    
    gpuRadixBitSort<<<numBlocks, THREADS_PER_BLOCK>>>(d_list, d_list2, l, r, d_divide, D);
    
    cudaMemcpy(d_list2, list, sizeof(data)*length, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_divide, divide, sizeof(int)*numBlocks, cudaMemcpyDeviceToHost);
    int nZeroes = 0;
    for(int i = 0; i < numBlocks; i++){
      nZeroes += divide[i];
    }
    D++;
    if(nZeroes > l)
      grSort(list, l, nZeroes - 1, length, D);
    if(nZeroes < r)
      grSort(list, nZeroes, r, length, D);
  }
  return;
}

double gpu_radixsort(float unsorted[], int length, float sorted[]){
  time_t start, stop;
  double time;

  data list[length];
  for(int i = 0; i< length; i++)
    list[i].val = (uint) unsorted[i];

  start = clock();
  grSort(list, 0, length - 1, length, 0);
  stop = clock();
  time = ((double) stop - start) / CLOCKS_PER_SEC;

  for(int j = 0; j < length; j++)
    sorted[j] = (float) list[j].val;

  return time;
}


void radixsort(float * unsorted, int length, Result * result){
  result = (Result *) malloc(sizeof(Result));
  if(result == NULL){
    fprintf(stderr, "Out of Memory\n");
    exit(1);
  }
  strcpy(result->tname, "Radix Sort");  
  float sorted[2][length];

  result->cpu_time = cpu_radixsort(unsorted, length, sorted[0]);
  result->gpu_time = gpu_radixsort(unsorted, length, sorted[1]);

  //check that sorted[0] = sorted[1];
  int n = 0;
  for(int i = 0; i < length; i++)
    if(sorted[0][i] != sorted[1][i])
      n++;
  if(n!= 0){
    fprintf(stderr, "There were %d discrepencies between the CPU and GPU Radix Sort algorithms\n", n);
  }

  return;

}
