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
#include "dnu/sp/floyd_warshall.h"
#include "dnu/sp/logger/Logger.h"

int main(int argc, char* argv[]) {
	string graphCsv = "A,A,0,A,B,4,A,D,5,A,E,5,B,B,0,B,C,7,B,D,3,C,C,0,C,F,4,D,A,7,D,C,3,D,D,0,D,E,4,D,F,3,E,A,2,E,D,6,E,E,0,F,D,2,F,E,1,F,F,0";
	string csvFilePath = "data/14edges.csv";
	string txt1000FilePath = "data/avias.txt";
	clock_t gpuTime1=clock();
	string shortnessPathGpuCsv = floydWarshallGpu(txt1000FilePath);
	clock_t gpuTime2=clock();
	clock_t cpuTime1=clock();
	string shortnessPathCpuCsv = floydWarshallCpu(txt1000FilePath);
	clock_t cpuTime2=clock();
//	INFO("Weight matrix csv from GPU: %s", shortnessPathGpuCsv.c_str());
//	INFO("Weight matrix csv from CPU: %s", shortnessPathCpuCsv.c_str());
	INFO("GPU and CPU solutions are the same: %s", (shortnessPathCpuCsv.compare(shortnessPathGpuCsv)==0)?"true":"false");
//	clock_t gpuExecutionTime = gpuTime2 - gpuTime1;
	double gpuSec = ((double)gpuTime2-gpuTime1)/CLOCKS_PER_SEC;
//	clock_t cpuExecutionTime = cpuTime2 - cpuTime1;
	double cpuSec = ((double)cpuTime2-cpuTime1)/CLOCKS_PER_SEC;
//	cout<<endl<<"GPU time: "<<double(gpuExecutionTime)/CLOCKS_PER_SEC<<endl;
	INFO("GPU execution time: %f sec", gpuSec);
	INFO("CPU execution time: %f sec", cpuSec);
	INFO("The difference between GPU and CPU is %f sec", (cpuSec - gpuSec));
	return 0;
}
