#ifndef FWHELPERS_H
#define FWHELPERS_H
#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
using namespace std;
int* readGraph(int&N, int&Na, char* argv);
void printArray(int Na, int* a);
#endif
