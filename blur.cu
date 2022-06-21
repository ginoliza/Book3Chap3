#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lodepng.h"

#define CHECK_ERROR(call) { 
	cudaError_t err = call; 
	if (err != cudaSuccess) { 
		printf("%s en %s linea %d\n", cudaGetErrorString(err), __FILE__, __LINE__); 
		exit(err); 
	} 
}

#define BLUR_SIZE 1 
#define CHANNEL 4


__global__
void blurKernel(unsigned char *Pin, unsigned char *Pout, int width, int height) {
    
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;
        
    if ( Col < width  && Row < height) {
    	int pixVal = 0;
    	int pixels = 0;        
    	
    	for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; blurRow++){
            
    		for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; blurCol++){
                
    			int curRow = Row + blurRow;
    			int curCol = Col + blurCol;    			
    			
    			if(curRow > -1 && curRow < height && curCol > -1 && curCol < width){
    				pixVal += Pin[curRow * width + curCol];
    				pixels++;
    			}
    		}
    	}        
        Pout[(Row * width + Col) * CHANNEL] = (unsigned char)(pixVal/pixels);
        Pout[(Row * width + Col) * CHANNEL + 1] = (unsigned char)(pixVal/pixels);
        Pout[(Row * width + Col) * CHANNEL + 2] = (unsigned char)(pixVal/pixels);
        Pout[(Row * width + Col) * CHANNEL + 3] = 255;
    }
}


void blur(unsigned char *h_Pin, unsigned char *h_Pout, int m, int n) {
    
    int size = (m*n*4)*sizeof(unsigned char);
    unsigned char *d_Pin, *d_Pout;    
    
    CHECK_ERROR(cudaMalloc((void**)&d_Pin, size));
    CHECK_ERROR(cudaMalloc((void**)&d_Pout, size));
        
    cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyHostToDevice);
        
    dim3 dimGrid(ceil(m / 16.0),ceil(n / 16.0), 1);
    dim3 dimBlock(16.0, 16.0, 1);
    blurKernel<<<dimGrid, dimBlock>>>(d_Pin, d_Pout, m, n);
        
    cudaMemcpy(h_Pout, d_Pout, size, cudaMemcpyDeviceToHost);
        
    cudaFree(d_Pin);
    cudaFree(d_Pout);
    
}

unsigned char* decodeOneStep(const char* filename)
{
    unsigned error;
    unsigned char* image;
    unsigned width, height;
    
    error = lodepng_decode32_file(&image, &width, &height, filename);
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
    
    return image;
}

void encodeOneStep(const char* filename, unsigned char* image, int width, int height)
{
    unsigned error = lodepng_encode32_file(filename, image, width, height);
 
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char *argv[]) {    
    
    if (argc != 2) {        
        exit(1);
    }
    const char *filename = argv[1];
        
    unsigned char *h_Pin, *h_Pout;
    
    int m = 512; 
    int n = 512;     
    
    h_Pin = (unsigned char*)malloc(sizeof(unsigned char)*(n*m));
    h_Pout = (unsigned char*)malloc(sizeof(unsigned char)*(n*m*4));
        
    
    h_Pin = decodeOneStep(filename);
    blur(h_Pin, h_Pout, m, n);    
        
    encodeOneStep("blurImage.png", h_Pout, m, n);
    
    free(h_Pin);
    free(h_Pout);
    
    return 0;
}