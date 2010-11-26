#include <math.h>
#include <stdio.h>
#include "../common/cuPrintf.cu"

#define SHAREDSIZE 8000  /* Should be changed to dynamically detect shared
                             memory size if at all possible.  */

//Forward declarations
__global__ void GPUMerge(float *d_list, int len, int stepSize,
                         int eltsPerThread);

/* Mergesort definition.  Takes a pointer to a list of floats.
 * the length of the list, the number of threads per block, and 
 * the number of blocks on which to execute.  
 * Puts the list into sorted order in-place.*/
void MergeSort(float *h_list, int len, int threadsPerBlock, int blocks) {

    float *d_list;
    if ( (cudaMalloc((void **) &d_list, len*sizeof(float))) == 
         cudaErrorMemoryAllocation) {
        printf("Error:  Insufficient device memory at line %d\n", __LINE__);
        return;
    }

    if (threadsPerBlock > 1024) {
        printf("ThreadsPerBlock must be no larger than 1024.");
        return;
    }

    cudaMemcpy(d_list, h_list, len*sizeof(float), cudaMemcpyHostToDevice);


    printf("Len is %d, tpb*blocks is %d\n", len, threadsPerBlock*blocks);
    int eltsPerBlock = (int)ceil(len/(float)blocks);
    printf("eltsPerBlock is %d\n", eltsPerBlock);
    
    int eltsPerThread = (int)ceil(eltsPerBlock/(float)threadsPerBlock);
    printf("eltsPerThread is %d\n", eltsPerThread);
    
    int maxStep = SHAREDSIZE/sizeof(float);

    if (maxStep < eltsPerBlock) {
        eltsPerBlock = maxStep;
    }

    printf("Attempting to call GPUMerge\n\n");
    GPUMerge<<<blocks, threadsPerBlock>>>(d_list, len, eltsPerBlock,
                                          eltsPerThread);

    cudaMemcpy(h_list, d_list, len*sizeof(float), cudaMemcpyDeviceToHost);

}

/* Mergesort definition.  Takes a pointer to a list of floats, the length
 * of the list, and the number of list elements given to each thread.
 * Puts the list into sorted order in-place.*/
__global__ void GPUMerge(float *d_list, int len, int eltsPerBlock,
                         int eltsPerThread){ //ENSURE EPT IS NOT TOO LARGE!

    int my_start, my_end; //indices of each thread's start/end in shared mem
    int g_my_start, g_my_end; //indices of thread's start/end in global mem    

    //Declare counters requierd for recursive mergesort
    int l_start, r_start; //Start index of the two lists being merged
    int walkLen; //Length of the left sublist being merged
    int old_l_start; 
    int l_end, r_end; //End index of the two lists being merged
    int headLoc; //current location of the write head on the newList
    short curList = 0; /* Will be used to determine which of two lists is the
                        * most up-to-date, since merge sort is not an in-place
                        * sorting algorithm. */

    //Attempt to allocate enough shared memory for this block's list...
    //Note that mergesort is not an in-place sort, so we need double memory.
    __shared__ float subList[2][SHAREDSIZE/sizeof(float)];

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

    //Local (shared memory) coordinates of start and end of list in this block
    my_start = g_my_start - blockIdx.x*eltsPerBlock;
    my_end = g_my_end - blockIdx.x*eltsPerBlock;

    //Load all memory for this block
    cuPrintf("g_my_start:  %d, eltsPerThread:  %d\n", g_my_start, eltsPerThread);
    for (int i = g_my_start, j = my_start; i < g_my_end; i++, j++){
        subList[curList][j] = d_list[i];
        cuPrintf("Copied memory at %d.  Value is %f.  Should be %f\n",
                 i, subList[curList][j], d_list[i]);
    }

    //Wait until all memory has been loaded
    __syncthreads();

    walkLen = 1;

    //tStride is the number of elements that a thread is responsible for.
    for (int tStride = eltsPerThread; tStride <= 2*eltsPerBlock; tStride *= 2){

        cuPrintf("   tStride:  %d.\n", tStride);

        my_start = tStride*threadIdx.x;
        if (my_start > eltsPerBlock){
            break;
        }

        my_end = my_start + tStride;

        //Did we overrun the block?
        if (my_end > eltsPerBlock){
            my_end = eltsPerBlock;
        }

        //Did we overrun the list?
        if (my_end + blockIdx.x*eltsPerBlock > len ){
            my_end = len - blockIdx.x*eltsPerBlock;
        }

        cuPrintf("   my_start:  %d.  my_end: %d.\n", my_start, my_end);

        //Merge the left and right lists.  Walklen is the current length
        //of the left sublist.
        for ( ; walkLen <= tStride; walkLen *= 2) { 
            
            cuPrintf("Walklen is now %d\n", walkLen);
            
            //Set up start and end indices.
            l_start = my_start;
            headLoc = l_start; //Write head starts here.
            
            while (l_start < my_end) {
                cuPrintf("curList is now %d\n", curList); 
                old_l_start = l_start; /*l_start will be incremented soon,
                                        *and we want to track where it began */
                cuPrintf("l_start is now %d\n", l_start);
                
                //If we reach the end of the list, we are done.
                if (l_start > my_end) {
                    l_start = my_end;
                }
                
                l_end = l_start + walkLen;
                if (l_end > my_end) {
                    l_end = my_end;
                    cuPrintf("l_end = my_end (%d)\n", l_end); 
                }
                
                r_start = l_end;
                cuPrintf("r_start = %d \n", r_start); 
                if (r_start > my_end) {
                    r_start = my_end;
                    cuPrintf("r_start = my_end (%d)\n", r_start); 
                }
                
                r_end = r_start + walkLen;
                if (r_end > my_end) {
                    r_end = my_end;
                    cuPrintf("r_end = my_end (%d)\n", r_end); 
                }
                
                //Perform the merge process
                while ((l_start < l_end) || (r_start < r_end)) {
                    cuPrintf("(l_start, end): (%d, %d),(r_start, end): (%d, %d)\n",
                             l_start, l_end, r_start, r_end);
                    
                    //Check if l is now empty
                    if (l_start == l_end) {
                        for (int j = r_start; j < r_end; j++){
                            subList[!curList][headLoc] = subList[curList][r_start];
                            cuPrintf("Writing %f from curlist[%d] to !curlist[%d] at %d\n",                            
                                     subList[curList][r_start], r_start,
                                     headLoc, __LINE__);
                            r_start++;
                            headLoc++;
                        }
                        break;
                    }
                    
                    //Check if r is empty
                    if (r_start == r_end) {
                        for (int j = l_start; j < l_end; j++){
                            subList[!curList][headLoc] = subList[curList][l_start];
                            cuPrintf("Writing %f from curlist[%d] to !curlist[%d] at %d\n",                                     
                                     subList[curList][l_start], l_start,
                                     headLoc, __LINE__);
                            l_start++;
                            headLoc++;
                        }
                        break;
                    }
                    
                    //Is left lead smaller than right lead?
                    if (subList[curList][l_start] < subList[curList][r_start]) {
                        subList[!curList][headLoc] = subList[curList][l_start];
                        cuPrintf("Writing %f from curlist[%d] to !curlist[%d] at %d\n",
                                 subList[curList][l_start], l_start, 
                                 headLoc, __LINE__);
                        l_start++;
                        headLoc++; 
                    }
                    
                    //Is right lead smaller than left lead?
                    else {
                        subList[!curList][headLoc] = subList[curList][r_start];
                        cuPrintf("Writing %f from curlist[%d] to !curlist[%d] at %d\n",
                                 subList[curList][r_start], r_start,
                                 headLoc, __LINE__);
                        r_start++;
                        headLoc++;
                    }
                }
                
                l_start = old_l_start + 2*walkLen;
            }
            curList = !curList;
        }
        walkLen = tStride;
        __syncthreads();
    }

    __syncthreads();

    //Local (shared memory) coordinates of start and end of list to write back
    my_start = g_my_start - blockIdx.x*eltsPerBlock;
    my_end = g_my_end - blockIdx.x*eltsPerBlock;

    //Write all memory for this block back to global memory
    cuPrintf("g_my_start:  %d, eltsPerThread:  %d\n", g_my_start, eltsPerThread);
    for (int i = g_my_start, j = my_start; i < g_my_end; i++, j++){
        d_list[i] = subList[curList][j];
        cuPrintf("Writing %f to d_list at %d.\n",
                 subList[curList][j], i);
    }

    __syncthreads();

    return;

    //subList[blockIdx

    //...otherwise, we use global memory...
    /*
    if ( (subList = cudaMalloc(eltsPerBlock*sizeof(float)) != NULL ) {
            //   do some shit.
            
        }    

    //...otherwise, we give up.
    */
}

void usage(){
    printf("Usage: in_list [thread_count] [kernel_count]"); 
}

int main(int argc, char *argv[]){
    
    int len;
    float *h_list;

    cudaPrintfInit();

    if ((argc > 4) || argc < 2) {
        printf("Invalid argument count.  %s accepts 1-4 arguments, %d given",
               argv[0], argc);
        usage();
        return -1;
    }
    
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    int thread_count = devProp.maxThreadsDim[0];
    //int kernel_count = devProp.maxGridSize[0];
    int kernel_count = 1;

    if (argc > 2){
        thread_count = atoi(argv[2]);
    }
    if (argc > 3){
        kernel_count = atoi(argv[3]);
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
    MergeSort(h_list, len, thread_count, kernel_count);

    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();

    printf("\nFinal list is:\n");
    for (int i = 0; i < len; i++){
        printf("%f\n", h_list[i]);
    }

    return 0;
}
