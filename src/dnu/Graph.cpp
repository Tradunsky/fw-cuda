#include "Graph.h"

int Graph::getVerticiesCount() {
	return N;
}

int Graph::getWeightMatrixWidth() {
	return Na;
}

void Graph::readGraphByLine(char* argv) {
// Read the graph file from memory
	string vname1, vname2;
	ifstream graphFile;
	string dummyString;
	int thisWeight; /* weight of the edge just read from file */
	N = 0; /* number of vertices */
	graphFile.open(argv);
//Read the graph into some maps
	graphFile >> vname1;
	while (vname1 != "--END--") {
		graphFile >> vname2;
		graphFile >> thisWeight;
		addEdge(vname1, vname2, thisWeight);
//		if (nameToNum.count(vname1) == 0) {
//			nameToNum[vname1] = N;
//			weightMap[vname1][vname1] = 0;
//			N++;
//		}
//		if (nameToNum.count(vname2) == 0) {
//			nameToNum[vname2] = N;
//			weightMap[vname2][vname2] = 0;
//			N++;
//		}
		weightMap[vname1][vname2] = thisWeight;
		graphFile >> vname1;
	}
	graphFile.close();
}

int* Graph::refreshWeightMatrix() {
	N = nameToNum.size();
	// "alignment" is what stored row sizes must be a multiple of
	Na = ALIGNMENT * ((N + ALIGNMENT - 1) / ALIGNMENT); /* for the sizes of our arrays */
	// Build the array
	weightMatrix = new int[N * Na];
	for (int ii = 0; ii < N; ii++)
		for (int jj = 0; jj < N; jj++)
			weightMatrix[ii * Na + jj] = __INT_MAX__;
	map<string, int>::iterator i;
	map<string, int>::iterator j;
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i) {
		for (j = nameToNum.begin(); j != nameToNum.end(); ++j) {
			if (weightMap[(*i).first].count((*j).first) != 0) {
				weightMatrix[Na * (*i).second + (*j).second] =
						weightMap[(*i).first][(*j).first];
			}
		}
	}
	return weightMatrix;
}

void Graph::readGraphByCsv(char* fileName) {
	ifstream fs(fileName);
	if (fs.is_open()) {
		string line;
		while (getline(fs, line)) {
			stringstream linestream(line);
			string vname1, vname2, weight;
			while (getline(linestream, vname1, ',') != NULL) {
				getline(linestream, vname2, ',');
				getline(linestream, weight, ',');
				int thisWeight = atoi(weight.c_str());
				//assign map of edges...
				addEdge(vname1, vname2, thisWeight);
			}
		}

		fs.close();
	}
}

void Graph::addEdge(string vName1, string vName2, int weight) {
	if (nameToNum.count(vName1) == 0) {
		nameToNum[vName1] = N;
		weightMap[vName1][vName1] = 0;
		N++;
	}
	if (nameToNum.count(vName2) == 0) {
		nameToNum[vName2] = N;
		weightMap[vName2][vName2] = 0;
		N++;
	}
	weightMap[vName1][vName2] = weight;
}

void Graph::printWeightMatrix() {
	map<string, int>::iterator i, j;
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i)
		if (i->second < 10)
			printf("\t%s", i->first.c_str());
	printf("\n");
	j = nameToNum.begin();
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i) {
		if (i->second < 10) {
			printf("%s\t", i->first.c_str());
			for (j = nameToNum.begin(); j != nameToNum.end(); ++j) {
				if (j->second < 10) {
					int distance = weightMatrix[i->second * Na + j->second];
					if (distance != __INT_MAX__)
						printf("%d\t", distance);
					else
						printf("--\t");
				}
			}
			printf("\n");
		}
	}
}

string Graph::toCsv() {
	map<string, map<string, int> >::iterator i;
	map<string, int>::iterator j;
	stringstream sstm;
	for (i = weightMap.begin(); i != weightMap.end(); ++i) {
		for (j = i->second.begin(); j != i->second.end(); ++j) {
			sstm << i->first << "," << j->first << "," << j->second << ",";
		}
	}
	return sstm.str();
}

string Graph::toWeightMatrixCsv() {
	map<string, int>::iterator i, j;
	stringstream sstm;
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i) {
		for (j = nameToNum.begin(); j != nameToNum.end(); ++j) {
			int distance = weightMatrix[i->second * Na + j->second];
			sstm << i->first << "," << j->first << ",";
			if (distance != __INT_MAX__)
				sstm << distance << ",";
			else
				sstm << "inf,";
		}
	}
	return sstm.str();
}

Graph::~Graph() {
	delete[] weightMatrix;
}

Graph::Graph(char* fileName) {
	readGraphByLine(fileName);
	refreshWeightMatrix();
}
//Graph(string csvGraph){}

int* Graph::toWeightMatrix() {
	return weightMatrix;
}
