#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common/io.h"
#include <time.h>
#include <cuda.h>


/************************** DATA DEFINITIONS *******************************/
// CONSTANTS
#define THREADS_PER_BLOCK 64
#define M 10

// MACROS
#define swap(A,B) { float temp = A; A = B; B = temp;}
#define compswap(A,B) if(B < A) swap(A,B)
#define digit(A,B) (((A) >> (8 - ((B)+1) * 8)) & ((1 << 8) - 1))
#define ch(A) digit(A, D)

// STRUCTS AND ENUM'S
typedef enum {
  BUCKET0 = 0,
  BUCKET1
}state;

typedef struct data{
  uint val;
  state bucket;
} data;

/**************************** CPU RADIX SORT *********************************/

/* Flip
 *
 * To perform radix sort on floating points, we want to convert them to
 * unsigned integers to perform bitwise comparisons. We will also need to 
 * flip the sign bit of positive floats (or flip all the other bits of negative
 * floats) to preserve ordering.
 *
 * Parameters:
 * list: The list of floating point values to be flipped
 * flipped: An output parameter. The values, having been converted, are stored
 *          here
 * length: The length of the list and flipped arrays.
 *
 */
void Flip(float list[], uint flipped[], int length){
  for(int i = 0; i < length; i++){
    uint temp = (uint) list[i];
    uint mask = -((int) ( temp >> 31)) | 0x80000000;
    flipped[i] = mask ^ ((uint) temp);
  }
  return;
}

/* unFlip
 *
 * After we have performed radix sort on the unsigned integers, we need to 
 * convert them back into floating point values via the reverse process used
 * to convert them into unsigned integeres with Flip.
 *
 * Parameters:
 * list: The list of unsigned integers to be converted back into floating
 *       points.
 * unFlipped: An output parameter, this is where the converted unsigned int's
 *            are stored after conversion.
 * length: The length of the arrays.
 *
 */
void unFlip(uint list[], float unFlipped[], int length){
  for(int i = 0; i< length; i++){
    uint temp = list[i];
    uint mask = ((temp >> 31) - 1) | 0x80000000;
    unFlipped[i] = (float) (mask ^ temp);
  }
  return;
}

/* insert
 *
 * Once the size of the list being sorted gets small enough, the overhead of
 * the radix quicksort implementation actually hampers performance, so we 
 * include an implementation of insertion sort to handle these short sublists.
 *
 * Parameters:
 * ls: The entire list of unsigned integers being sorted by Radix Quick Sort
 * l: The left bound of the section of ls being sorted by insert
 * r: The right bound of the section of ls being sorted by insert
 *
 */
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

/* RadixQuicksort
 *
 * This is an implementation of the Radix Quicksort algorithm described in 
 * 'Algorithms in C' by Robert Sedgewick, Program 10.3 (page 422).
 *
 * Parameters:
 * ls: The list of unsigned integers being sorted.
 * l: The left bound of the section of ls being operated on in this call to
 *    RadixQuicksort.
 * r: The right bound of the section of ls being operated on in this call to
 *    RadixQuicksort.
 * D: The radix currently being compared, that is the index of the bit 
 *    (valued from 0 to 31) by which elements of ls are currently being 
 *    sorted by.
 */
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

/* cpu_radixsort
 *
 * This is the wrapper function around RadixQuicksort, the purpose of which
 * is to set up the floating point array (convert it to unsigned integers),
 * set up the timing, call RadixQuicksort and then convert the unsigned
 * integers back into floating points.
 *
 * Parameters:
 * unsorted: The list of floating points to be sorted.
 * length: The length of the arrays
 * sorted: An output parameter, contains the list of floating points after the
 *         sorting algorithm has been executed.
 *
 * Return Value:
 * time: This function returns the time of execution of the sorting algorithm
 *       as a double precision floating point.
 */
double cpu_radixsort(float unsorted[], int length, float sorted[]){
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


/**************************** GPU RADIX SORT ********************************/

/* devFlip
 *
 * As with the CPU implementation of radix sort, we need to convert the
 * floating point values to unsigned ints, though now we can exploit the
 * inherently data parallel nature of this process.
 *
 * Parameters:
 * u: This is merely the floating point not yet properly converted, merely cast
 *    as an unsigned integer.
 * D: This is the radix (index of the bit being compared). We only want to
 *    perform this flip once, that is before the first comparison (when D = 0).
 *
 * Return Value:
 * u': This function returns either u (if D != 0), or the properly bit-flipped
 *     value of u.
 */
__device__ uint devFlip(uint u, int D){
  if( !D ){
    uint mask = -((int) ( u >> 31)) | 0x80000000;
    return mask ^ u;
  }
  return u;
}

/* devUnFlip
 *
 * This function performs the 'unflipping' procedure as was described in the
 * CPU function unFlip. It exploits the inherently parallel nature of this
 * process.
 *
 * Parameters:
 * u: The unsigned integer to be 'unflipped'
 * D: The radix. We only want to perform this unflip procedure after the final
 *    bitwise comparison, so we check that (D == 31).
 *
 * Return Value:
 * u': This function returns the 'unflipped' value of u if D == 31, otherwise
 *     it simply returns u.
 */
__device__ float devUnFlip(uint u, int D){
  if(D == 31){
     uint mask = ((u >>31) - 1) | 0x80000000;
     return (float) (mask ^ u);
  }
  return u;
}

/* gpuRadixBitSort
 *
 * This kernel function implements the comparison and partition portion of an
 * implementation of Radix Quicksort.
 *
 * Parameters:
 * input: The input array of data structs to be sorted
 * output: An output parameter, will be populated with the partitioned/
 *         partially ordered list of data structs. As input, is identical
 *         to the input[] array, but the region [l,r] will be partitioned
 *         properly by the function.
 * l: The index of the left bound of the input being sorted.
 * r: The index of the right bound of the input being sorted.
 * nZeroes: An output parameter as well as counting struct. Each thread block
 *          keeps a counter of the number of values with a '0' at position D,
 *          this is used by the Host (after the function returns) to calculate
 *          where the partition between 0 and 1 occurs to recursively call
 *          gpuRadixBitSort.
 * D: The Radix, ranging from 0 to 31, this keeps track of which bit is being 
 *    compared.
 */
__global__ void gpuRadixBitSort(data input[], data output[], int l, int r, 
				int nZeroes[], int D)
{
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

/* grSort
 *
 * This function performs the recursive GPU Radix quick sorting by calculating 
 * the number of thread blocks requires, calling gpuRadixBitSort and 
 * recursively calling itself while ranging over the Radix value (0 to 31).
 *
 * Parameters:
 * ls: The list of data structs being sorted.
 * l: The left bound index on the section of ls being sorted currently.
 * r: The right bound index on the section of ls being sorted currently.
 * length: The length of ls.
 * D: The Radix, the index ranging from 0 to 31 corresponding to the bit being
 *    sorted at the moment.
 */
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
    
    gpuRadixBitSort<<<numBlocks, THREADS_PER_BLOCK>>>(d_list, d_list2, l, r, 
						      d_divide, D);
    
    cudaMemcpy(d_list2, list, sizeof(data)*length, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_divide, divide, sizeof(int)*numBlocks, 
	       cudaMemcpyDeviceToHost);
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

/* gpu_radixsort
 *
 * This function is a wrapper around the call to grSort, meant to handle the
 * conversions to and from a floating point array to a data struct array, as
 * well as the timing functions to measure the speed of the actual sorting 
 * algorithm.
 *
 * Parameters:
 * unsorted: The list of floating point values to be sorted
 * length: The length of the arrays
 * sorted: An output parameter, will contain the results of applying the
 *         sorting algorithm.
 *
 * Return Value:
 * time: This function returns the the time of execution of the gpu radix
 *       sorting algorithm as a double-precision floating point.
 */
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

/* radixsort
 * 
 * This function makes calls to the CPU and GPU implementations of Radix Sort
 * and populates a Result Struct. It also performs a quick check to ensure
 * that the results of each sorting algorithm are consistent (a debugging
 * feature).
 *
 * Parameters:
 * unsorted: A list of floating points to be sorted
 * length: The length of the unsorted array
 * result: An output parameter to be populated with the name of the test and
 *         the times of execution of the CPU and GPU implementations of Radix
 *         sort.
 */
void radixsort(float unsorted[], int length, Result * result){
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
