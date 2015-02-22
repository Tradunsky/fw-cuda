#include "floyd_warshall.h"
#include "logger/Logger.h"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/*******************************************
 * Kernel to improve the shortest paths by considering the kth vertex
 * as an intermediate.
 *
 * arg: k = the index of the vertex we are using as intermediate
 * arg: weightMatrix = the array containing the matrix of weight
 * arg: verticeCount = the number of vertices in the graph
 * arg: matrixWidth = the width of the matrix as stored on the device’s memory.
 *
 * The graph is stored as an verticeCount x matrixWidth matrix, with the (i, j) matrix entry
 * being stored as weightMatrix[i*matrixWidth + j]
 *
 *******************************************/
__global__ void fwStepK(int k, int weightMatrix[], int matrixWidth,
		int verticeCount) {
	/* This thread’s matrix column */
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < verticeCount) {
		int currentEdge = matrixWidth * blockIdx.y + col;
		__shared__ int trkc;
		/* this row, kth column */
		// Improve by using the intermediate k, if we can
		if (threadIdx.x == 0)
			trkc = weightMatrix[matrixWidth * blockIdx.y + k];
		__syncthreads();
		if (trkc != __INT_MAX__) {
			/* this column, kth row */
			int tckr = weightMatrix[k * matrixWidth + col];
			if (tckr != __INT_MAX__) {
				int intermediateWeight = trkc + tckr;
				if (intermediateWeight < weightMatrix[currentEdge])
					weightMatrix[currentEdge] = intermediateWeight;
			}
		}
	}
}

//util
bool matrixIsEquals(int* matrixA, int* matrixB, int length, int width) {
	bool isEquals = true;
	for (int i = 0; i < length && isEquals; i++) {
		for (int j = 0; j < length && isEquals; j++) {
			if (matrixA[i * width + j] != matrixB[i * width + j]) {
				isEquals = false;
			}
		}
	}
	return isEquals;
}

Graph readGraph(string filePath) {
	Graph graph;
	const char* fileName = filePath.c_str();
	string fileExtension = filePath.substr(filePath.find_last_of(".") + 1);
	if (fileExtension == "csv") {
//		Logger::LOG(Level::INFO, "Reading graph from csv file: %s", fileName);
		INFO("Reading graph from csv file: %s", fileName);
		graph.readFromCsvFile(fileName);
	} else if (fileExtension == "txt") {
		INFO("Reading graph from txt file: %s", fileName);
		graph.readFromTextFile(fileName);
	} else {
		//it's csv graph
		INFO("Reading graph by csv string: %s", fileName);
		graph.addCsvGraph(filePath);
	}
	return graph;
}
//algs

string floydWarshallGpu(string filePath) {
	Graph graph = readGraph(filePath);
	DEBUG("Original graph csv for GPU: %s", graph.toCsv().c_str());
	int* hostWeightMatrix = graph.toWeightMatrix();
	int* deviceWeightMatrix;
	// Number of vertices
	int verticeCount = graph.getVerticiesCount();
	// Width of matrix to encourage coalescing
	int matrixWidth = graph.getWeightMatrixWidth();
	int weightMatrixSize = matrixWidth * verticeCount * sizeof(int);
	// Copy the array to newly-allocated global memory
	CUDA_CHECK_RETURN(cudaMalloc(&deviceWeightMatrix, weightMatrixSize));
	CUDA_CHECK_RETURN(
			cudaMemcpy(deviceWeightMatrix, hostWeightMatrix, weightMatrixSize,
					cudaMemcpyHostToDevice));
	// Set up and run the kernels
	//TODO: DETECT OR SPECIFY threadsPerBlock BY CUDA API
	int threadsPerBlock = 256;
	dim3 blocksPerGrid((matrixWidth + threadsPerBlock - 1) / threadsPerBlock,
			verticeCount);
	// The kth run through this loop considers whether we might do better using
	// the kth vertex as an intermediate
	for (int k = 0; k < verticeCount; k++) {
		fwStepK<<<blocksPerGrid, threadsPerBlock>>>(k, deviceWeightMatrix,
				matrixWidth, verticeCount);
		CUDA_CHECK_RETURN(cudaThreadSynchronize());
	}
	//TODO: copy result to new pointer.
	//Result will copy to graph weight matrix...
	CUDA_CHECK_RETURN(
			cudaMemcpy(hostWeightMatrix, deviceWeightMatrix, weightMatrixSize,
					cudaMemcpyDeviceToHost));

	string csvWieghtMatrix = graph.toWeightMatrixCsv();
	CUDA_CHECK_RETURN(cudaFree(deviceWeightMatrix));
//	if (NULL!=hostWeightMatrix){
//		free(hostWeightMatrix);
//		hostWeightMatrix = NULL;
//	}
	return csvWieghtMatrix;
}

string floydWarshallCpu(string filePath) {
	Graph graph = readGraph(filePath);
	DEBUG("Original graph csv for CPU: %s", graph.toCsv().c_str());
	int verticeCount = graph.getVerticiesCount();
	int weightMatrixWidth = graph.getWeightMatrixWidth();
	int* weightMatrix = graph.toWeightMatrix();
	for (int k = 0; k < verticeCount; k++) {
		for (int i = 0; i < verticeCount; i++) {
			for (int j = 0; j < verticeCount; j++) {
				int kc = weightMatrix[i * weightMatrixWidth + k];
				if (kc != __INT_MAX__) {
					int kr = weightMatrix[k * weightMatrixWidth + j];
					if (kr != __INT_MAX__) {
						int intermediateWeight = kc + kr;
						int currentEdge = i * weightMatrixWidth + j;
						if (intermediateWeight < weightMatrix[currentEdge]) {
							weightMatrix[currentEdge] = intermediateWeight;
						}
					}
				}
			}
		}
	}
//	bool matrixIsSame = matrixIsEquals(weightMatrix, graph.toWeightMatrix(), verticeCount, weightMatrixWidth);
//	printf("\nCPU weight matrix is the same as graph weight matrix: %s", matrixIsSame?"true":"false");
	return graph.toWeightMatrixCsv();
}
