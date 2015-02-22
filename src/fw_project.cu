/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
//#include "dnu/graph/Graph.h"
#include "dnu/sp/floyd_warshall.h"

///*******************************************
// * Kernel to improve the shortest paths by considering the kth vertex
// * as an intermediate.
// *
// * arg: k = the index of the vertex we are using as intermediate
// * arg: weightMatrix = the array containing the matrix of weight
// * arg: verticeCount = the number of vertices in the graph
// * arg: matrixWidth = the width of the matrix as stored on the device’s memory.
// *
// * The graph is stored as an N x Na matrix, with the (i, j) matrix entry
// * being stored as devArray[i*Na + j]
// *
// *******************************************/
//__global__ void fwStepK(int k, int weightMatrix[], int matrixWidth, int verticeCount) {
//	/* This thread’s matrix column */
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	if (col >= verticeCount)
//		return;
//	int arrayIndex = matrixWidth * blockIdx.y + col;
//	__shared__ int trkc;
//	/* this row, kth column */
//// Improve by using the intermediate k, if we can
//	if (threadIdx.x == 0)
//		trkc = weightMatrix[matrixWidth * blockIdx.y + k];
//	__syncthreads();
//	if (trkc == __INT_MAX__)
//		/* infinity */
//		return;
//	/* this column, kth row */
//	int tckr = weightMatrix[k * matrixWidth + col];
//	if (tckr == __INT_MAX__)
//		/* infinity */
//		return;
//	int betterMaybe = trkc + tckr;
//	if (betterMaybe < weightMatrix[arrayIndex])
//		weightMatrix[arrayIndex] = betterMaybe;
//}

//int floydWarshall();

int main(int argc, char* argv[]) {
	string shortnessPathGpuCsv = floydWarshallGpu("data/14edges.csv");
	string shortnessPathCpuCsv = floydWarshallCpu("data/14edges.csv");
	printf("\nWeight matrix csv from GPU: %s", shortnessPathGpuCsv.c_str());
	printf("\nWeight matrix csv from CPU: %s", shortnessPathCpuCsv.c_str());
	printf("\nGPU and CPU solutions are the same: %s", (shortnessPathCpuCsv.compare(shortnessPathGpuCsv)==0)?"true":"false");
	return 0;
}

//int floydWarshall() {
//	char* textFilePath = "data/14edges.txt";
//	char* csvFilePath = "data/14edges.csv";
//	Graph graph(textFilePath);
////	graph.readFromCsvFile(csvFilePath);
//	int* weightMatrix = graph.toWeightMatrix();
//	int N = graph.getVerticiesCount(); /* Number of vertices*/
//	int Na = graph.getWeightMatrixWidth(); /* Width of matrix to encourage coalescing */
//	printf("Kernel: Just read %s with %d vertices, Na = %d\n", textFilePath, N, Na);
//	// Copy the array to newly-allocated global memory
//	int* devArray;
//	cudaError_t err = cudaMalloc(&devArray, Na * N * sizeof(int));
//	printf("Malloc device rules: %s\n", cudaGetErrorString(err));
//	err = cudaMemcpy(devArray, weightMatrix, Na * N * sizeof(int),
//			cudaMemcpyHostToDevice);
//	printf("Pre-kernel copy memory onto device: %s\n", cudaGetErrorString(err));
//	// Set up and run the kernels
//	int threadsPerBlock = 256;
//	dim3 blocksPerGrid((Na + threadsPerBlock - 1) / threadsPerBlock, N);
//	// The kth run through this loop considers whether we might do better using
//	// the kth vertex as an intermediate
//	for (int k = 0; k < N; k++) {
//		fwStepK<<<blocksPerGrid, threadsPerBlock>>>(k, devArray, Na, N);
//		err = cudaThreadSynchronize();
//		// Uncomment the following line when debugging the kernel
//		// printf("Kernel: using %d as intermediate: error = %s\n", k, cudaGetErrorString
//		// Uncomment the following two lines to print intermediate results
//		// err = cudaMemcpy(graph, devArray, Na*N*sizeof(int), cudaMemcpyDeviceToHost);
//		// printArray(Na, graph);
//	}
//	err = cudaMemcpy(weightMatrix, devArray, Na * N * sizeof(int),
//			cudaMemcpyDeviceToHost);
//	printf("Post-kernel copy memory off of device: %s\n",
//			cudaGetErrorString(err));
//	graph.printWeightMatrix();
//	string csvGraph = graph.toCsv();
//	printf("\nGraph csv: %s", csvGraph.c_str());
//	string csvWeightMatrix = graph.toWeightMatrixCsv();
//	printf("\nWeight matrix csv: %s", csvWeightMatrix.c_str());
//	cudaFree(devArray);
//	return 0;
//}
