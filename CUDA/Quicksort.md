
# Quicksort

    cdpSimpleQuicksort.cu - GPU 파일
    SimpleQuicksort.cpp   - CPU 파일
    
    Parameters :
        v : 리스트에 어떤 값이 들어가는지 보여줌(지정안함 추천)
        num_items=NUMBER : 정렬 대상의 개수 지정(필수)
        device=DEVICE_NUM : GPU 대상 옵션으로 CUDA를 돌릴 GPU 선택(GPU 기준 미선택시 자동 0번 할당)


```python
%%file cdpSimpleQuicksort.cu

/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#include <iostream>
#include <cstdio>
#include <ctime>
#include "../../common/inc/helper_cuda.h"
#include "../../common/inc/helper_string.h"

#define MAX_DEPTH       16
#define INSERTION_SORT  32

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(unsigned int *data, int left, int right)
{
    for (int i = left ; i <= right ; ++i)
    {
        unsigned min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            unsigned val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right, int depth)
{
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)
    {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *lptr = data+left;
    unsigned int *rptr = data+right;
    unsigned int  pivot = data[(left+right)/2];

    // Do the partitioning.
    while (lptr <= rptr)
    {
        // Find the next left- and right-hand values to swap
        unsigned int lval = *lptr;
        unsigned int rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot)
        {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot)
        {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(unsigned int *data, unsigned int nitems)
{
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // Launch on device
    int left = 0;
    int right = nitems-1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
    checkCudaErrors(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void initialize_data(unsigned int *dst, unsigned int nitems)
{
    // Fixed seed for illustration
    srand(2047);

    // Fill dst with random values
    for (unsigned i = 0 ; i < nitems ; i++)
        dst[i] = rand() % nitems ;
}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
void check_results(int n, unsigned int *results_d)
{
    unsigned int *results_h = new unsigned[n];
    checkCudaErrors(cudaMemcpy(results_h, results_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int i = 1 ; i < n ; ++i)
        if (results_h[i-1] > results_h[i])
        {
            std::cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << std::endl;
            exit(EXIT_FAILURE);
        }

    std::cout << "OK" << std::endl;
    delete[] results_h;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int num_items = 128;
    bool verbose = false;

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h"))
    {
        std::cerr << "Usage: " << argv[0] << " num_items=<num_items>\twhere num_items is the number of items to sort" << std::endl;
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "v"))
    {
        verbose = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "num_items"))
    {
        num_items = getCmdLineArgumentInt(argc, (const char **)argv, "num_items");

        if (num_items < 1)
        {
            std::cerr << "ERROR: num_items has to be greater than 1" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Find/set device and get device properties
    int device = -1;
    cudaDeviceProp deviceProp;
    device = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));

    if (!(deviceProp.major > 3 || (deviceProp.major == 3 && deviceProp.minor >= 5)))
    {
        printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting.", device, deviceProp.name);
        exit(EXIT_WAIVED);
    }

    // Create input data
    clock_t start = clock();
    unsigned int *h_data = 0;
    unsigned int *d_data = 0;

    // Allocate CPU memory and initialize data.
    std::cout << "Initializing data:" << std::endl;
    h_data =(unsigned int *)malloc(num_items*sizeof(unsigned int));
    initialize_data(h_data, num_items);

    if (verbose)
    {
        for (int i=0 ; i<num_items ; i++)
            std::cout << "Data [" << i << "]: " << h_data[i] << std::endl;
    }

    // Allocate GPU memory.
    checkCudaErrors(cudaMalloc((void **)&d_data, num_items * sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Execute
    std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
    run_qsort(d_data, num_items);

    // Check result
    std::cout << "Validating results: ";
    check_results(num_items, d_data);

    free(h_data);
    checkCudaErrors(cudaFree(d_data));

    std::cout << "During time: " << ( ( double )clock() - start ) / CLOCKS_PER_SEC << " (s)" << std::endl;

    exit(EXIT_SUCCESS);
}


```

    Overwriting cdpSimpleQuicksort.cu



```python
!make
```

    "/usr/local/cuda"/bin/nvcc -ccbin g++ -I../../common/inc  -m64    -dc -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 -o cdpSimpleQuicksort.o -c cdpSimpleQuicksort.cu
    "/usr/local/cuda"/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 -o cdpSimpleQuicksort cdpSimpleQuicksort.o  -lcudadevrt
    mkdir -p ../../bin/x86_64/linux/release
    cp cdpSimpleQuicksort ../../bin/x86_64/linux/release



```python
!./cdpSimpleQuicksort num_items=1000000 device=3
```

    CUDA error at ../../common/inc/helper_cuda.h:1078 code=38(cudaErrorNoDevice) "cudaGetDeviceCount(&device_count)" 



```python
%%file SimpleQuicksort.cpp

#include <iostream>
#include <cstdio>
#include <ctime>
#include "../../common/inc/helper_cuda.h"
#include "../../common/inc/helper_string.h"

clock_t start_tick, end_tick;

void initialize_data( unsigned int *dst, unsigned int nitems )
{
    // Fixed seed for illustration
    srand( 2047 );

    // Fill dst with random values
    for ( unsigned i = 0; i < nitems; i++ )
        dst[ i ] = rand() % nitems;
}

void quickSort( unsigned int *array, int low, int high )
{
    int i = low;
    int j = high;
    int pivot = array[ ( i + j ) / 2 ];
    int temp;

    while ( i <= j )
    {
        while ( array[ i ] < pivot )
            i++;
        while ( array[ j ] > pivot )
            j--;
        if ( i <= j )
        {
            temp = array[ i ];
            array[ i ] = array[ j ];
            array[ j ] = temp;
            i++;
            j--;
        }
    }
    if ( j > low )
        quickSort( array, low, j );
    if ( i < high )
        quickSort( array, i, high );
}

void run_qsort( unsigned int *data, unsigned int nitems )
{
    // Launch on device
    int left = 0;
    int right = nitems - 1;
    
    quickSort( data, left, right );
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char **argv )
{
    int num_items = 128;
    bool verbose = false;

    if ( checkCmdLineFlag( argc, ( const char ** )argv, "help" ) ||
        checkCmdLineFlag( argc, ( const char ** )argv, "h" ) )
    {
        std::cerr << "Usage: " << argv[ 0 ] << " num_items=<num_items>\twhere num_items is the number of items to sort" << std::endl;
        exit( EXIT_SUCCESS );
    }

    if ( checkCmdLineFlag( argc, ( const char ** )argv, "v" ) )
    {
        verbose = true;
    }

    if ( checkCmdLineFlag( argc, ( const char ** )argv, "num_items" ) )
    {
        num_items = getCmdLineArgumentInt( argc, ( const char ** )argv, "num_items" );

        if ( num_items < 1 )
        {
            std::cerr << "ERROR: num_items has to be greater than 1" << std::endl;
            exit( EXIT_FAILURE );
        }
    }

    std::cout << "Running on CPU " << std::endl;
    start_tick = clock();

    // Create input data
    unsigned int *h_data = 0;

    // Allocate CPU memory and initialize data.
    std::cout << "Initializing data:" << std::endl;
    h_data = ( unsigned int * )malloc( num_items * sizeof( unsigned int ) );
    initialize_data( h_data, num_items );

    if ( verbose )
    {
        for ( int i = 0; i<num_items; i++ )
        std::cout << "Data [" << i << "]: " << h_data[ i ] << std::endl;
    }

    // Execute
    std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
    run_qsort( h_data, num_items );

    // Check result
    std::cout << "During time: " << ((double) clock() - start) / CLOCKS_PER_SEC << " (s)" << std::endl;

    // Finish
    free( h_data );
    exit( EXIT_SUCCESS );
}

```

    Overwriting SimpleQuicksort.cpp



```python
!g++ -Wall -o SimpleQuicksort SimpleQuicksort.cpp
```

    [01m[KSimpleQuicksort.cpp:[m[K In function '[01m[Kvoid quickSort(unsigned int*, int, int)[m[K':
    [01m[KSimpleQuicksort.cpp:29:28:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [-Wsign-compare]
             while ( array[ i ] < pivot )
    [01;32m[K                            ^[m[K
    [01m[KSimpleQuicksort.cpp:31:28:[m[K [01;35m[Kwarning: [m[Kcomparison between signed and unsigned integer expressions [-Wsign-compare]
             while ( array[ j ] > pivot )
    [01;32m[K                            ^[m[K
    [01m[KSimpleQuicksort.cpp:[m[K In function '[01m[Kint main(int, char**)[m[K':
    [01m[KSimpleQuicksort.cpp:110:57:[m[K [01;31m[Kerror: [m[K'[01m[Kstart[m[K' was not declared in this scope
         std::cout << "During time: " << ((double) clock() - start) / CLOCKS_PER_SEC
    [01;32m[K                                                         ^[m[K



```python
!./SimpleQuicksort num_items=1000000
```

    Running on CPU 
    Initializing data:
    Running quicksort on 1000000 elements
    During time :0.23982 (s)

