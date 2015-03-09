/*
 * floyd_warshall.h
 *
 *  Created on: 20 февр. 2015
 *      Author: o111o1oo
 */

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#ifndef FLOYD_WARSHALL_H_
#define FLOYD_WARSHALL_H_
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <map>
#include <string>
#include "graph/Graph.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>


using namespace std;

string floydWarshallGpu(string filePath);
string floydWarshallCpu(string filePath);

#endif /* FLOYD_WARSHALL_H_ */
