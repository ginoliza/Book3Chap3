#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CHECK_ERROR(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(err); \
	} \
}

// Cada thread realiza una suma por pares
// ejecutada en el device, llamable solo desde el host
__global__
void matrixAddKernel(float *d_Mout, float *d_Min1, float *d_Min2, int rows, int cols) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	
	// SOLUCION B: cada thread produce un elemento de matriz como salida
	if (i < rows && j < cols) {
		d_Mout[i*rows + j] = d_Min1[i*rows + j] + d_Min2[i*rows + j];
	}
	
	// SOLUCION C: cada thread produce una fila de la matriz como salida
	// if (j == 0 && i < rows) {
	// 	while (j < cols) {
	// 		d_Mout[i*rows + j] = d_Min1[i*rows + j] + d_Min2[i*rows + j];
	// 		j++;
	// 	}
	// }

	// SOLUCION D: cada thread produce una columna de la matriz como salida
	// if (i == 0 && j < cols) {
	// 	while (i < rows) {
	// 		d_Mout[i*rows + j] = d_Min1[i*rows + j] + d_Min2[i*rows + j];
	// 		i++;
	// 	}
	// }
}

// Calcular la suma del vector h_C = h_A + h_B
void matrixAdd(float *h_Mout, float *h_Min1, float *h_Min2, int rows, int cols) {

	float *d_Mout;
	float *d_Min1;
	float *d_Min2;
	int size = rows*cols *sizeof(float);

	CHECK_ERROR(cudaMalloc((void**)&d_Mout, size));
	CHECK_ERROR(cudaMalloc((void**)&d_Min1, size));
	CHECK_ERROR(cudaMalloc((void**)&d_Min2, size));

	cudaMemcpy(d_Min1, h_Min1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Min2, h_Min2, size, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(rows / 32.0), ceil(cols / 32.0), 1);
	dim3 dimBlock(32.0, 32.0, 1);

	matrixAddKernel<<<dimGrid, dimBlock>>>(d_Mout, d_Min1, d_Min2, rows, cols);

	cudaMemcpy(h_Mout, d_Mout, size, cudaMemcpyDeviceToHost);

	cudaFree(d_Mout);
	cudaFree(d_Min1);
	cudaFree(d_Min2);
}

int main(void) {

	float *h_Min1, *h_Min2, *h_Mout;
	int rows = 100;
	int cols = 100;

	h_Min1 = (float*)malloc(sizeof(float)*rows*cols);
	h_Min2 = (float*)malloc(sizeof(float)*rows*cols);
	h_Mout = (float*)malloc(sizeof(float)*rows*cols);
	
	// llenar Min1 y Min2 con float aleatorios
	srand(time(NULL));
	for (int i = 0; i < rows ; i++) {
		for (int j = 0; j < cols ; j++) {
			h_Min1[i*rows+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
			h_Min2[i*rows+j] = ((((float)rand() / (float)(RAND_MAX)) * 10));
		}
	}

	// suma de matrices
	matrixAdd(h_Mout, h_Min1, h_Min2, rows, cols);

	// verificar suma
	int valueIsCorrect = 1;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (h_Mout[i*rows + j] != (h_Min1[i*rows + j] + h_Min2[i*rows + j])) {
				printf("ERROR DE SUMA!\n");
				valueIsCorrect = 0;
			}
		}
	}
	if (valueIsCorrect) {
		printf("SUMA VERIFICADA\n");
	}

	free(h_Min1);
	free(h_Min2);
	free(h_Mout);

	return 0;
}
