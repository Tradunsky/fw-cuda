// shortness_path_exe.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "shortness_path.h"
#include <iostream>
#include <string>
#include <time.h>
#include <conio.h>

using namespace std;
using namespace SP;

int _tmain(int argc, _TCHAR* argv[]){
	string graphCsv;
	cout << "Enter a graph csv: ";
	cin >> graphCsv;
	cout << "Computing...";
	clock_t gpuTime1 = clock();
	string shortnessPathGpuCsv = FloydWarshall::csvFwGpu(graphCsv);
	clock_t gpuTime2 = clock();
	double gpuSec = ((double)gpuTime2 - gpuTime1) / CLOCKS_PER_SEC;

	clock_t cpuTime1 = clock();
	string shortnessPathCpuCsv = FloydWarshall::csvFwCpu(graphCsv);
	clock_t cpuTime2 = clock();
	double cpuSec = ((double)cpuTime2 - cpuTime1) / CLOCKS_PER_SEC;	

	cout << endl <<"Weight matrix csv from GPU: " << shortnessPathGpuCsv;
	//cout << "Weight matrix csv from CPU: " << shortnessPathCpuCsv;
	cout << endl << "GPU execution time: " << gpuSec << " sec.";
	cout << endl << "CPU execution time: " << cpuSec << " sec.";
	cout << endl << "The difference between GPU and CPU is " << (cpuSec - gpuSec) << " sec";
	string isTheSameSolutions = (shortnessPathCpuCsv.compare(shortnessPathGpuCsv) == 0) ? "true" : "false";
	cout << endl << "GPU and CPU solutions are the same: " << isTheSameSolutions;

	_getch();
	return 0;
}

