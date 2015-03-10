#ifndef SHORTNESS_PATH_H
#define SHORTNESS_PATH_H
//#ifndef __CUDACC__  
//#define __CUDACC__
//#endif
//#ifdef SPDLL_EXPORTS
//#define SPDLL_API __declspec(dllexport) 
//#else
//#define SPDLL_API __declspec(dllimport) 
//#endif
//#include "stdafx.h"
#include <stdexcept>
#include "Graph.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>

using namespace std;

namespace SP{
	class FloydWarshall{
	public:
		static __declspec(dllexport) string csvFwGpu(string csvGraph);
		static __declspec(dllexport) string csvFwCpu(string csvGraph);
	};
}
#endif