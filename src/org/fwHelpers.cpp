#include "fwHelpers.h"

map<string, int> nameToNum; /* names of vertices */
map<string, map<string, int> > weightMap; /* weights of edges */
int* readGraph(int& N, int& Na, char* argv) {
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
		if (nameToNum.count(vname1) == 0) {
			nameToNum[vname1] = N;
			weightMap[vname1][vname1] = 0;
			N++;
		}
		if (nameToNum.count(vname2) == 0) {
			nameToNum[vname2] = N;
			weightMap[vname2][vname2] = 0;
			N++;
		}
		weightMap[vname1][vname2] = thisWeight;
		graphFile >> vname1;
	}
	graphFile.close(); // Nice and Tidy
// "alignment" is what stored row sizes must be a multiple of
	int alignment = ALIGNMENT;
	Na = alignment * ((N + alignment - 1) / alignment); /* for the sizes of our arrays */
	printf("Alignment = %d\n", alignment);
// Build the array
	int* a = (int*) malloc(N * Na * sizeof(int));
	for (int ii = 0; ii < N; ii++)
		for (int jj = 0; jj < N; jj++)
			a[ii * Na + jj] = __INT_MAX__;
	map<string, int>::iterator i;
	map<string, int>::iterator j;
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i)
		for (j = nameToNum.begin(); j != nameToNum.end(); ++j) {
			if (weightMap[(*i).first].count((*j).first) != 0) {
				a[Na * (*i).second + (*j).second] =
						weightMap[(*i).first][(*j).first];
			}
		}
	return a;
}
void printArray(int Na, int* a) {
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
					int dd = a[i->second * Na + j->second];
					if (dd != __INT_MAX__)
						printf("%d\t", dd);
					else
						printf("--\t");
				}
			}
			printf("\n");
		}
	}
}
