#include <math.h>
#include <stdio.h>
#include "../common/cuPrintf.cu"

#define BLOCKSIZE 1023  //Size of blocks at the bottom heap
#define OUTSIZE 512 //Size of output shared memory
#define BLOCKDEPTH 10 //Max Depth of bottom heap, and ceil of log of blocksize

//Tells us our current progress on building a given block.
typedef struct blockInfo{
    short bufsize;
    short writeloc;
    int heapified; //Only a bool is needed, but int will maintain alignment.
} blockInfo_t;


//Forward declarations
__global__ void GPUHeapSort(float *d_list, blockInfo_t *blockInfo,
                            int len, int topHeapSize,
                            int botHeapSize, 
                            int warpSize, int metaDepth);
__device__ void bottomLevel(float *d_list, int len); //NYI
__device__ void topLevel(float *d_list, int len); //NYI
__device__ void heapify(__volatile__ float *in_list, int len, 
                        __volatile__ int *temp);
__device__ void pipelinedPop(__volatile__ float *heap, float *out_list, 
                             int d, int popCount,
                             __volatile__ int *temp);
__device__ void loadBlock(float *g_block, float *s_block, int blockLen,
                          blockInfo_t *g_info, blockInfo_t *s_info);
__device__ void writeBlock(float *g_block, __volatile__ float *s_block,
                           int blockLen);
__device__ void printBlock(float *s_block, int blockLen);
__host__ int heapSort(float *h_list, 
                      int len, int threadsPerBlock,
                      int blocks, cudaDeviceProp devProp);
__host__ int floorlog2(int x);

//Ceiling of log2 of x.  Could be made faster, but effect would be negligible.
int ceilLog2(int x){
    if (x < 1){
        return -1;
    }
    x--;
    int output = 0;
    while (x > 0) {
        x >>= 1;
        output++;
    }
    return output;
}

/* Heapsort definition.  Takes a pointer to a list of floats.
 * the length of the list, the number of threads per block, and 
 * the number of blocks on which to execute.  
 * Puts the list into sorted order in-place.*/
int heapSort(float *h_list, int len, int threadsPerBlock, int blocks,
              cudaDeviceProp devProp) {

    float *d_list;
    blockInfo_t *blockInfo;
    int logLen; //log of length of list
    int metaDepth; //layers of metaheaps
    int topHeapSize; //Size of the top heap
    int logBotHeapSize; //log_2 of max size of the bottom heaps 
    int logMidHeapSize; //log_2 of max size of intermediate heaps
    int temp;

    //Trivial list?  Just return.
    if (len < 2){
        return 0;
    }

    //Ensure that we have a valid number of threads per block.
    if (threadsPerBlock == 0){
        threadsPerBlock = devProp.maxThreadsPerBlock;
    }
    //We require a minimum of 2 warps per block to run our code
    else if (threadsPerBlock < 2*devProp.warpSize){
        threadsPerBlock = 64;
    }
    if (threadsPerBlock > devProp.maxThreadsPerBlock) {
        printf("Device cannot handle %d threads per block.  Max is %d",
               threadsPerBlock, devProp.maxThreadsPerBlock);
        return -1;
    }
     
    //Calculate size of heaps.  BotHeapSize is 1/8 shared mem size.
    //logBotHeapSize = ceilLog2(devProp.sharedMemPerBlock>>3);
    logBotHeapSize = BLOCKDEPTH;
    logMidHeapSize = logBotHeapSize - 2;

    printf("logBotHeap: %d, logMidHeap: %d\n", logBotHeapSize, logMidHeapSize);

    //Calculate metaDepth and topHeapSize.
    metaDepth = 0; //Will increment this if necessary.
    logLen = ceilLog2(len);
    temp = logBotHeapSize; //temp is a counter tracking total subheap depth.
    
    //Do we only need one heap?
    if (temp >= logLen){
        topHeapSize = len;
    }
    //Otherwise, how many metaheaps do we need?
    else {
        while (temp < logLen){
            metaDepth++;
            temp += logMidHeapSize;
        }
        topHeapSize = len>>temp;
    }

    printf("metaDepth is %d\n", metaDepth);
    printf("topHeapSize is %d\n", topHeapSize); 

    if (metaDepth > blocks){
        printf("Must have at least metaDepth blocks available.");
        printf("metaDepth is %d, but only %d blocks were given.\n", 
               metaDepth, blocks);
        return -1;
    }

    if (metaDepth > 2){
        printf("Current implementation only supports metaDepth of 2.  ");
        printf("Given metadepth was %d.  In practice, ", metaDepth); 
        printf("this means that list lengths cannot equal or exceed 2^20.");
    }


    if ( (cudaMalloc((void **) &d_list, len*sizeof(float))) == 
         cudaErrorMemoryAllocation) {
        printf("Error:  Insufficient device memory at line %d\n", __LINE__);
        return -1;
    }

    cudaMemcpy(d_list, h_list, len*sizeof(float), cudaMemcpyHostToDevice);
    
    if ( (cudaMalloc((void **) &blockInfo, len*sizeof(blockInfo_t))) == 
         cudaErrorMemoryAllocation) {
        printf("Error:  Insufficient device memory at line %d\n", __LINE__);
        return -1;
    }
    
    printf("Attempting to call GPUHeapSort\n\n");
    /*
    GPUHeapSort<<<blocks, threadsPerBlock,
        ( 1<<(logBotHeapSize + 3) + 1<<(logBotHeapSize+2) ) >>>
        (d_list, blockInfo, len, topHeapSize, 1<<logBotHeapSize, 
         devProp.warpSize, metaDepth);
    */
    GPUHeapSort<<<blocks, threadsPerBlock>>>
        (d_list, blockInfo, len, 0, BLOCKSIZE, devProp.warpSize, metaDepth);

    cudaThreadSynchronize();
    cudaMemcpy(h_list, d_list, len*sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/* GPUHeapSort definition.  Takes a pointer to a list of floats, the length
 * of the list, and the number of list elements given to each thread.
 * Puts the list into sorted order in-place.*/
__global__ void GPUHeapSort(float *d_list, blockInfo_t *blockInfo,
                            int len, int topHeapSize,
                            int botHeapSize,
                            int warpSize, int metaDepth){
    
    __shared__ __volatile__ float heap[BLOCKSIZE];
    __shared__ float output[OUTSIZE];
    __shared__ blockInfo_t curBlockInfo;
    __shared__ int g_start, g_end;
    __shared__ int blockLen;
    __shared__ __volatile__ int temp; //temporay variable for device functions.

    int i = 512;
    if (i > len) {
        i = len;
    }

    g_start = blockIdx.x*botHeapSize;
    g_end = (blockIdx.x+1)*botHeapSize;    

    if (g_end > len){
        g_end = len;
    }
    
    blockLen = g_end-g_start;
    //Load memory
    
    loadBlock(&d_list[g_start], (float *)heap, blockLen,
              &blockInfo[blockIdx.x], &curBlockInfo);
    
    __syncthreads();
    
    
    //printBlock((float *) heap, blockLen);
    
    //First warp heapifies
    if (threadIdx.x < 8){
        heapify(heap, blockLen, &temp);
    }
    __syncthreads();

    //printBlock((float *)heap, i);

    //First warp pops
    if (threadIdx.x < 8){
        pipelinedPop(heap, (float *)output, BLOCKDEPTH, i, &temp);
        //__threadfence_block();
    }

    __syncthreads();
    
    writeBlock(&d_list[g_start], heap, blockLen);
    
    printBlock((float *)output, i);

    return;
}

/* Loads a block of data from global memory into shared memory.  Must be
 * called by all threads of a thread block to ensure proper operation. 
 */
__device__ void loadBlock(float *g_block, float *s_block, int blockLen,
                          blockInfo_t *g_info, blockInfo_t *s_info){

    for(int i = threadIdx.x; i < blockLen; i += blockDim.x){
        s_block[i] = g_block[i];
        g_block[i] = s_block[i];
    }
    if (threadIdx.x == 0){
        *s_info = *g_info;
    }
}

/* Writes a block of data from shared memory into global memory.  Must be
 * called by all threads of a thread block to ensure proper operation. 
 */
__device__ void writeBlock(float *g_block,
                           __volatile__ float *s_block, int blockLen){

    for(int i = threadIdx.x; i < blockLen; i += blockDim.x){
        //cuPrintf("Writing %f at location %d\n", s_block[i], i);
        //g_block[i] = s_block[i];
    }
}

/* Prints a block of data in shared memory */
__device__ void printBlock(float *s_block, int blockLen){
    for (int i = threadIdx.x; i < blockLen; i += blockDim.x){
        cuPrintf("s_block[%d] = %f\n", i, s_block[i]);
    }
}

/* Heapifies a list using a single warp.  Must be run on the bottom warp of a
 * thread.  If this function is not executed by all of threads 0-7, the GPU
 * will stall.
 */
__device__ void heapify(__volatile__ float *inList, int len,
                        __volatile__ int *temp){
    
    int focusIdx = 0; //Index of element currently being heapified
    float focus=0, parent=0; //current element being heapified and its parent
    /*int localTemp=0; Temp doesn't need to be re-read _every_ time.
                    * Temp will be used to track the next element to percolate.
                    */

    if (threadIdx.x == 0){
        *temp = 0; //Index of next element to heapify
    }

    //localTemp = 0;
    
    //We maintain the invariant that no two threads are processing on
    //adjacent layers of the heap in order to avoid memory conflicts and
    //race conditions.
    while (*temp < len){
        if (threadIdx.x == (*temp & 7)){
            focusIdx = *temp;
            focus = inList[focusIdx];
            *temp = *temp + 1;
            //cuPrintf("Focusing on element %d with value %f\n",
            //         focusIdx, focus);
        }
        
        //Unrolled loop once to avoid race conditions and get a small speed
        //boost over using a for loop on 2 iterations.
        if (focusIdx != 0){
            parent = inList[(focusIdx-1)>>1];
            //Swap focus and parent if focus is bigger than parent
            if (focus > parent){
                //cuPrintf("Focus %f > parent %f\n", focus, parent); 
                inList[focusIdx] = parent;
                inList[(focusIdx-1)>>1] = focus;
                focusIdx = (focusIdx - 1)>>1;
            }
            else {
                //cuPrintf("Parent %f > focus %f\n", parent, focus);
                focusIdx = 0;
            }
        }
        if (focusIdx != 0){
            parent = inList[(focusIdx-1)>>1];
            //Swap focus and parent if focus is bigger than parent
            if (focus > parent){
                //cuPrintf("Focus %f > parent %f\n", focus, parent); 
                inList[focusIdx] = parent;
                inList[(focusIdx-1)>>1] = focus;
                focusIdx = (focusIdx-1)>>1;
            }
            else {
                //cuPrintf("Parent %f > focus %f\n", parent, focus);
                focusIdx = 0; 
            }
       }
        //localTemp = *temp;
    }
    return;
}

/* Pops a heap using a single warp.  Must be run on the bottom warp of a
 * thread.  If this function is not executed by all of threads 0-7, the GPU
 * will stall.
 * heap: a pointer to a heap structure w/ space for a complete heap of depth d 
 * d:  The depth of the heap 
 * count: The number of elements to pop
 */
__device__ void pipelinedPop(__volatile__ float *heap, float *out_list, 
                             int d, int popCount,
                             __volatile__ int *temp){
    
    int focusIdx = 0; //Index of element currently percolating down
    int maxChildIdx=0; //Index of largest child of element percolating down
    int curDepth=d+1; //Depth of element currently percolating down
    /*int localTemp=0; Temp doesn't need to be re-read _every_ time.
                    * Temp will be used to track the next element to percolate.
                    */

    if (threadIdx.x == 0){
        *temp = 0; //We have thus far popped 0 elements
    }

    //localTemp = 0;
    
    //We maintain the invariant that no two threads are processing on
    //adjacent layers of the heap in order to avoid memory conflicts and
    //race conditions.
    while (*temp < popCount){
        if (threadIdx.x == (*temp & 7)){
            focusIdx = 0;
            curDepth = 0;
            out_list[*temp] = heap[0];
            *temp = *temp + 1;
            //cuPrintf("temp is: %d\n", *temp);
            //cuPrintf("top of heap is: %f\n", heap[0]);
        }
        
        //Unrolled loop once to avoid race conditions and get a small speed
        //boost over using a for loop on 2 iterations.
        if (curDepth < d-1){
            maxChildIdx = 2*focusIdx+1;
            //cuPrintf("Children are %f, %f\n", heap[2*focusIdx+2], 
            //         heap[maxChildIdx]); 
            //cuPrintf("Depth is %d, Focusing on element %d\n", curDepth,
            //         focusIdx);
            if (heap[2*focusIdx+2] > heap[maxChildIdx]){
                maxChildIdx = 2*focusIdx+2;
            }
            heap[focusIdx] = heap[maxChildIdx];
            focusIdx = maxChildIdx;
            curDepth++;
        }

        if (curDepth < d-1){
            maxChildIdx = 2*focusIdx+1;
            //cuPrintf("Depth is %d, Focusing on element %d\n", curDepth,
            //         focusIdx);
            if (heap[2*focusIdx+2] > heap[maxChildIdx]){
                maxChildIdx = 2*focusIdx+2;
            }
            heap[focusIdx] = heap[maxChildIdx];
            focusIdx = maxChildIdx;
            curDepth++;
        }

        if (curDepth == d-1){
            //cuPrintf("curDepth is %d\n", curDepth);
            //cuPrintf("focusIdx is %d\n", focusIdx);
            //cuPrintf("Depth is %d (max).  Focusing on element %d\n", curDepth,
            //focusIdx);
            heap[focusIdx] = 0;
            curDepth++;
            //continue;
        }
    }
    
    //empty the pipeline before returning
    
    while (curDepth < d-1){
        //cuPrintf("Emptying Pipeline.  Focusing on element %d\n", focusIdx); 
        maxChildIdx = 2*focusIdx+1;
        if (heap[2*focusIdx+2] > heap[maxChildIdx]){
            maxChildIdx = 2*focusIdx+2;
        }
        heap[focusIdx] = heap[maxChildIdx];
        focusIdx = maxChildIdx;
        curDepth++;
    }
    

    return;
}

void usage(){
    printf("Usage: in_list [thread_count] [kernel_count]\n"); 
}

int main(int argc, char *argv[]){
    
    int len;
    float *h_list;

    cudaPrintfInit();

    if ((argc > 4) || argc < 2) {
        printf("Invalid argument count.  %s accepts 1-4 arguments, %d given\n",
               argv[0], argc);
        usage();
        return -1;
    }
    
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    int thread_count = 1;
    //int block_count = devProp.maxGridSize[0];
    int block_count = 1;

    if (argc > 2){
        thread_count = atoi(argv[2]);
    }
    if (argc > 3){
        block_count = atoi(argv[3]);
    }

    FILE *fin = fopen(argv[1], "r");
    
    if (fin == NULL){
        printf("Could not open file: %s", argv[1]);
        return -2;
    }

    fscanf(fin, "%d", &len);

    h_list = (float *)malloc(len*sizeof(float));
    if (h_list == NULL){
        printf("Insufficient host memory to allocate at %d", __LINE__);
        return -3;
    }

    for (int i = 0; i < len; i++){
        if (EOF == fscanf(fin, "%f ", &h_list[i])){
            break;
        }
    }

    
    printf("\nInitial list is:\n");
    for (int i = 0; i < len; i++){
        printf("%f\n", h_list[i]);
    }
    

    //MergeSort(h_list, len, devProp.maxThreadsDim[0], devProp.maxGridSize[0]);
    //MergeSort(h_list, len, devProp.maxThreadsDim[0], 1);
    heapSort(h_list, len, thread_count, block_count, devProp);

    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();

    
    printf("\nFinal list is:\n");
    for (int i = 0; i < len; i++){
        printf("%f\n", h_list[i]);
    }
    

    return 0;
}
