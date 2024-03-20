#include <stdio.h>

__global__ void hello(){
	printf("hello GPU\n");
}

int main(void){
	hello<<<4,4>>>();
	cudaDeviceSynchronize();

	return 0;
}
