#include <math.h>
#include <stdio.h>
#include "../common/cuPrintf.cu"

//Forward declarations
__global__ void GPUHeapSort(float *d_list, int len, int topHeapSize,
                            int botHeapSize, int eltsPerThread,
                            int warpSize, int metaDepth);
__device__ void bottomLevel(float *d_list, int len); //NYI
__device__ void topLevel(float *d_list, int len); //NYI
__device__ void heapify(float *in_list, int len);
__device__ void loadBlock(float *block, int blockLen);
__host__ int heapSort(float *h_list, int len, int threadsPerBlock,
                      int blocks, cudaDeviceProp devProp);
__host__ int floorlog2(int x);

typedef struct blockInfo{
    short bufsize;
    short writeloc;
    int heapified; //Only a bool is needed, but int will maintain alignment.
} blockInfo_t;

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
    int eltsPerThread; /* Count of elements that a given thread must handle.
                          on the bottom heap. */
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
        printf("Device cannot handle %d threads per block.\
max is %d",
               threadsPerBlock, devProp.maxThreadsPerBlock);
        return -1;
    }
     
    //Calculate size of heaps.  BotHeapSize is 1/8 shared mem size.
    logBotHeapSize = ceilLog2(devProp.sharedMemPerBlock>>3);
    logMidHeapSize = logBotHeapSize - 2;

    printf("logBotHeap: %d, logMidHeap: %d\n", logBotHeapSize, logMidHeapSize);

    if (logBotHeapSize == 0 || logMidHeapSize == 0){
        printf("Error:  Insufficient GPU shared memory to run heapSort");
        return -1;
    }

    //Calculate eltsPerThread
    eltsPerThread = (int)ceil(float(1<<logBotHeapSize)/float(threadsPerBlock));

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
    GPUHeapSort<<<blocks, threadsPerBlock,
        1<<(logBotHeapSize + 3) + 1<<(logBotHeapSize+2) >>>
        (d_list, len, topHeapSize, 1<<logBotHeapSize, eltsPerThread, 
         devProp.warpSize, metaDepth);

    cudaMemcpy(h_list, d_list, len*sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}

/* GPUHeapSort definition.  Takes a pointer to a list of floats, the length
 * of the list, and the number of list elements given to each thread.
 * Puts the list into sorted order in-place.*/
__global__ void GPUHeapSort(float *d_list, blockInfo_t *blockInfo,
                            int len, int topHeapSize,
                            int botHeapSize, int threadsPerBlock,
                            int warpSize, int metaDepth){

    extern __shared__ float array[];
    float *heap = array;
    float *output = &array[botHeapSize];
    
    int my_start, my_end; //indices of each thread's start/end in shared mem
    int g_my_start, g_my_end; //indices of thread's start/end in global mem    
    

    cuPrintf("Considering copying memory\n");
    //Load memory

    //Global coordinates of start and end of lists
    g_my_start = blockIdx.x*eltsPerBlock+threadIdx.x*eltsPerThread;
    g_my_end=blockIdx.x*eltsPerBlock+(threadIdx.x+1)*eltsPerThread;
    
    //Did we run past the end of the block?
    if (g_my_end > (blockIdx.x+1)*eltsPerBlock){
        g_my_end = (blockIdx.x+1)*eltsPerBlock;
    }

    //Did we run past the end of the list?
    if (g_my_end > len){
        g_my_end = len;
    }
    /*
    //Local (shared memory) coordinates of start and end of list in this block
    my_start = g_my_start - blockIdx.x*eltsPerBlock;
    my_end = g_my_end - blockIdx.x*eltsPerBlock;

    //Load all memory for this block
    cuPrintf("g_my_start:  %d, eltsPerThread:  %d\n", g_my_start, eltsPerThread);
    for (int i = g_my_start, j = my_start; i < g_my_end; i++, j++){
        subList[j] = d_list[i];
        //cuPrintf("Copied memory at %d.  Value is %f.  Should be %f\n",
        //         i, subList[j], d_list[i]);
    }

    //Wait until all memory has been loaded
    __syncthreads();
    */
    return;
}

/* Loads a block of data from global memory into shared memory.  Must be
 * called by all threads of a thread block to ensure proper operation. 
 */
__device__ void loadBlock(float *g_block, float *l_block, int blockLen){

    for(int i = threadIdx.x; i < blockLen; i *= blockDim.x){
        l_block[i] = g_block[i];
    }
}

/* Heapifies a list using a single warp.  Must be run on the bottom warp of a
 * thread.  If this function is not executed by all of threads 0-7, the GPU
 * will stall.
 */
__device__ void heapify(__volatile__ float *inList, int len, int depth){
    int curTailIdx = 0; //Index of next element to heapify
    int focusIdx; //Index of element currently being heapified
    float focus, parent; //current element being heapified and its parent

    //We maintain the invariant that no two threads are processing on
    //adjacent layers of the heap in order to avoid memory conflicts and
    //race conditions.
    while (curTailIdx < len){
        if (threadIdx.x == (curTailIdx && 8)){
            focusIdx = curTailIdx;
            focus = inList[focusIdx];
            curTailIdx++;
        }

        //Unrolled loop once to avoid race conditions and get a small speed
        //boost over using a for loop on 2 iterations.
        if (focusIdx != 0){
            parent = inList[focusIdx>>1];
            //Swap focus and parent if focus is bigger than parent
            if (focus > parent){
                inList[focusIdx] = parent;
                inList[focusIdx>>1] = focus;
                focusIdx >>= 1;
            }
        }
        if (focusIdx != 0){
            parent = inList[focusIdx>>1];
            //Swap focus and parent if focus is bigger than parent
            if (focus > parent){
                inList[focusIdx] = parent;
                inList[focusIdx>>1] = focus;
                focusIdx >>= 1;
            }
        }
    }
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

    int thread_count = devProp.maxThreadsDim[0];
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
