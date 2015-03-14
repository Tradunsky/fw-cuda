#ifndef GRAPH_H
#define GRAPH_H
#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif
#ifndef CSV_SEPARATOR
#define CSV_SEPARATOR ';'
#endif
//#include "stdafx.h"
#include <string>
#include <map>
#include <fstream>
#include <sstream>

using namespace std;

class Graph {
private:
	map<string, int> nameToNum;
	map<string, map<string, double> > weightMap;
	int N = 0, Na = 0;
	double* weightMatrix = NULL;
	bool needRefreshWeightMatrix = false;
	bool readGraphByLine(const char* fileName);
	bool readGraphByCsv(const char* fileName);
	double* refreshWeightMatrix();
public:
	void addEdge(string vName1, string vName2, double weight);
	void addCsvGraph(string line);
	double* toWeightMatrix();
	int getVerticiesCount();
	int getWeightMatrixWidth();
	string toCsv();
	string toWeightMatrixCsv();
	wstring toWWeightMatrixCsv();
	void printWeightMatrix();
	void clear();
	bool readFromTextFile(const char* filePath);
	bool readFromCsvFile(const char* filePath);
	~Graph();
	//Graph(string filePath);
	Graph(){
		N = 0;
		Na = 0;
		//		weightMatrix = NULL;
	};
};
#endif