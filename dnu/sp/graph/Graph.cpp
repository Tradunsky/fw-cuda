#include "Graph.h"

int Graph::getVerticiesCount() {
	if (needRefreshWeightMatrix)
		refreshWeightMatrix();
	return N;
}

int Graph::getWeightMatrixWidth() {
	if (needRefreshWeightMatrix)
		refreshWeightMatrix();
	return Na;
}

bool Graph::readGraphByLine(const char* argv) {
// Read the graph file from memory
	string vname1, vname2;
	ifstream graphFile;
	string dummyString;
	int thisWeight; /* weight of the edge just read from file */
	N = 0; /* number of vertices */	
	graphFile.open(argv);	
	if (graphFile.is_open()) {
		clear();
//Read the graph into some maps
		graphFile >> vname1;
		while (!graphFile.eof() && vname1 != "--END--") {
			graphFile >> vname2;
			graphFile >> thisWeight;
			addEdge(vname1, vname2, thisWeight);
//			weightMap[vname1][vname2] = thisWeight;
			graphFile >> vname1;
		}
		graphFile.close();
		return true;
	}
	return false;
}

int* Graph::refreshWeightMatrix() {
//	N = nameToNum.size();
	// "alignment" is what stored row sizes must be a multiple of
	Na = ALIGNMENT * ((N + ALIGNMENT - 1) / ALIGNMENT); /* for the sizes of our arrays */
	if (NULL != weightMatrix) {
		delete[] weightMatrix;
		weightMatrix = NULL;
	}
	// Build the array
	weightMatrix = new int[N * Na];
	for (int ii = 0; ii < N; ii++)
		for (int jj = 0; jj < N; jj++)
			weightMatrix[ii * Na + jj] = INT_MAX;
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
	needRefreshWeightMatrix = false;
	return weightMatrix;
}

bool Graph::readGraphByCsv(const char* fileName) {
	ifstream fs(fileName);
	if (fs.is_open()) {
		clear();
		string line;
		while (getline(fs, line)) {
			addCsvGraph(line);
		}
		fs.close();
		return true;
	}
	return false;
}

bool Graph::readFromTextFile(const char* filePath) {
	if (readGraphByLine(filePath)){
		refreshWeightMatrix();
		return true;
	}
	return false;
}
bool Graph::readFromCsvFile(const char* filePath) {
	if (readGraphByCsv(filePath)){
		refreshWeightMatrix();
		return true;
	}
	return false;
}

void Graph::addEdge(string vName1, string vName2, int weight) {
	int countOfName1 = nameToNum.count(vName1);
	int countOfName2 = nameToNum.count(vName2);
	if (countOfName1 == 0) {
		nameToNum[vName1] = N;
		weightMap[vName1][vName1] = 0;
		N++;
	}
	if (countOfName2 == 0) {
		nameToNum[vName2] = N;
		weightMap[vName2][vName2] = 0;
		N++;
	}
	weightMap[vName1][vName2] = weight;
	if (!needRefreshWeightMatrix)
		needRefreshWeightMatrix = true;
}

void Graph::addCsvGraph(string line) {
	stringstream linestream(line);
	string vname1, vname2, weight;
	while (getline(linestream, vname1, ',')) {
		getline(linestream, vname2, ',');
		getline(linestream, weight, ',');
		int thisWeight = atoi(weight.c_str());
		//assign map of edges...
		addEdge(vname1, vname2, thisWeight);
	}
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
					if (distance != INT_MAX)
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
			if (distance != INT_MAX)
				sstm << distance << ",";
			else
				sstm << "inf,";
		}
	}
	return sstm.str();
}

Graph::~Graph() {
	clear();
}

Graph::Graph(string filePath) {
	weightMatrix = NULL;
	N = 0;
	Na = 0;
//	const char* fileName = filePath.c_str();
//	if (filePath.substr(filePath.find_last_of(".") + 1) == "csv") {
//		readGraphByCsv(fileName);
//	} else {
//		readGraphByLine(fileName);
//	}
//	refreshWeightMatrix();
}
//Graph::Graph(string csvGraph) {
//	addCsvGraph(csvGraph);
//}

int* Graph::toWeightMatrix() {
	if (needRefreshWeightMatrix)
		refreshWeightMatrix();
	return weightMatrix;
}

void Graph::clear() {
	N = 0;
	Na = 0;
	if (!nameToNum.empty())
		nameToNum.clear();
	if (!weightMap.empty())
		weightMap.clear();
	if (NULL != weightMatrix) {
		delete weightMatrix;
		weightMatrix = NULL;
	}
}
