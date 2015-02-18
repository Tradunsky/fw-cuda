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
#include "org/fwHelpers.h"
#include "dnu/Graph.h"

/*******************************************
 * Kernel to improve the shortest paths by considering the kth vertex
 * as an intermediate.
 *
 * arg: k = the index of the vertex we are using as intermediate
 * arg: devArray = the array containing the matrix
 * arg: N = the number of vertices in the graph
 * arg: Na = the width of the matrix as stored on the device’s memory.
 *
 * The graph is stored as an N x Na matrix, with the (i, j) matrix entry
 * being stored as devArray[i*Na + j]
 *
 *******************************************/
__global__ void fwStepK(int k, int devArray[], int Na, int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x; /* This thread’s matrix column */
	if (col >= N)
		return;
	int arrayIndex = Na * blockIdx.y + col;
	__shared__ int trkc;
	/* this row, kth column */
// Improve by using the intermediate k, if we can
	if (threadIdx.x == 0)
		trkc = devArray[Na * blockIdx.y + k];
	__syncthreads();
	if (trkc == __INT_MAX__)
		return;
	/* infinity */
	int tckr = devArray[k * Na + col]; /* this column, kth row */
	if (tckr == __INT_MAX__)
		/* infinity */
		return;
	int betterMaybe = trkc + tckr;
	if (betterMaybe < devArray[arrayIndex])
		devArray[arrayIndex] = betterMaybe;
}
/******************************************************
 * main
 *
 * We read in the graph from a graph file, in this format:
 * First we read in N, the number of vertices
 * The vertices are assumed to be numbered 1..N
 * Then we read in triples of the form s d w
 *
 where s is the source vertex, d the destination, and w the weight
 *
 * The "best weight so far" matrix stores rows contiguously, with
 * bwsf(i, j) being the best weight FROM i TO j
 *
 * Thread blocks comprise rows of the matrix, so that we can
 * take advantage of global memory access grouping
 ********************************************************/
int runfwHelpers();
int runGraph();

int main(int argc, char* argv[]) {
	return runGraph();
}

int runGraph() {
	char* fileName = "data/14edges.csv";
	Graph graph(fileName);
	int* weightMatrix = graph.toWeightMatrix();
	int N = graph.getVerticiesCount(); /* Number of vertices*/
	int Na = graph.getWeightMatrixWidth(); /* Width of matrix to encourage coalescing */
	printf("Kernel: Just read %s with %d vertices, Na = %d\n", fileName, N, Na);
	// Copy the array to newly-allocated global memory
	int* devArray;
	cudaError_t err = cudaMalloc(&devArray, Na * N * sizeof(int));
	printf("Malloc device rules: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(devArray, weightMatrix, Na * N * sizeof(int),
			cudaMemcpyHostToDevice);
	printf("Pre-kernel copy memory onto device: %s\n", cudaGetErrorString(err));
	// Set up and run the kernels
	int threadsPerBlock = 256;
	dim3 blocksPerGrid((Na + threadsPerBlock - 1) / threadsPerBlock, N);
	// The kth run through this loop considers whether we might do better using
	// the kth vertex as an intermediate
	for (int k = 0; k < N; k++) {
		fwStepK<<<blocksPerGrid, threadsPerBlock>>>(k, devArray, Na, N);
		err = cudaThreadSynchronize();
		// Uncomment the following line when debugging the kernel
		// printf("Kernel: using %d as intermediate: error = %s\n", k, cudaGetErrorString
		// Uncomment the following two lines to print intermediate results
		// err = cudaMemcpy(graph, devArray, Na*N*sizeof(int), cudaMemcpyDeviceToHost);
		// printArray(Na, graph);
	}
	err = cudaMemcpy(weightMatrix, devArray, Na * N * sizeof(int),
			cudaMemcpyDeviceToHost);
	printf("Post-kernel copy memory off of device: %s\n",
			cudaGetErrorString(err));
	graph.printWeightMatrix();
	string csvGraph = graph.toCsv();
	printf("\nGraph csv: %s", csvGraph.c_str());
	string csvWeightMatrix = graph.toWeightMatrixCsv();
	printf("\nWeight matrix csv: %s", csvWeightMatrix.c_str());
	cudaFree(devArray);
	return 0;
}

int runfwHelpers() {
	char* fileName = "data/14edges.txt";
	int N = 0; /* Number of vertices
	 */
	int Na = 0; /* Width of matrix to encourage coalescing */
	int* graph = readGraph(N, Na, fileName); /* from fwHelpers.cpp
	 */
	printf("Kernel: Just read %s with %d vertices, Na = %d\n", fileName, N, Na);
	// Copy the array to newly-allocated global memory
	int* devArray;
	cudaError_t err = cudaMalloc(&devArray, Na * N * sizeof(int));
	printf("Malloc device rules: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(devArray, graph, Na * N * sizeof(int),
			cudaMemcpyHostToDevice);
	printf("Pre-kernel copy memory onto device: %s\n", cudaGetErrorString(err));
	// Set up and run the kernels
	int threadsPerBlock = 256;
	dim3 blocksPerGrid((Na + threadsPerBlock - 1) / threadsPerBlock, N);
	// The kth run through this loop considers whether we might do better using
	// the kth vertex as an intermediate
	for (int k = 0; k < N; k++) {
		fwStepK<<<blocksPerGrid, threadsPerBlock>>>(k, devArray, Na, N);
		err = cudaThreadSynchronize();
		// Uncomment the following line when debugging the kernel
		// printf("Kernel: using %d as intermediate: error = %s\n", k, cudaGetErrorString
		// Uncomment the following two lines to print intermediate results
		// err = cudaMemcpy(graph, devArray, Na*N*sizeof(int), cudaMemcpyDeviceToHost);
		// printArray(Na, graph);
	}
	err = cudaMemcpy(graph, devArray, Na * N * sizeof(int),
			cudaMemcpyDeviceToHost);
	printf("Post-kernel copy memory off of device: %s\n",
			cudaGetErrorString(err));
	printArray(Na, graph);
	free(graph);
	cudaFree(devArray);
	return 0;
}
