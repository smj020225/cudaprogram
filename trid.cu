#include <stdio.h>

__global__ void hello(){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int id = tid + bid * blockDim.x;

    printf("Hello world from block %d and thread %d,global id %d\n",bid, tid, id);
}

int main(void){
    hello<<<2,4>>>();
    cudaDeviceSynchronize();

    return 0;
}