/*
 * Graph.h
 *
 *  Created on: 17 февр. 2015
 *      Author: o111o1oo
 */

#ifndef GRAPH_H_
#define GRAPH_H_
#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

class Graph {
private:
	map<string, int> nameToNum;
	map<string, map<string, int> > weightMap;
	int N, Na;
	int* weightMatrix = NULL;
	void readGraphByLine(char* fileName);
	void readGraphByCsv(char* fileName);
	int* refreshWeightMatrix();
public:
	void addEdge(string vName1, string vName2, int weight);
	void addCsvGraph(string line);
	int* toWeightMatrix();
	int getVerticiesCount();
	int getWeightMatrixWidth();
	string toCsv();
	string toWeightMatrixCsv();
	void printWeightMatrix();
	void clear();
	void readFromTextFile(char* filePath);
	void readFromCsvFile(char* filePath);
	~Graph();
	Graph(char* fileName);
	Graph(){};
	Graph(string csvGraph);
};

#endif /* GRAPH_H_ */
