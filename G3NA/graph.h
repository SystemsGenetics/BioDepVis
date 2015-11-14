#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include "util.h"


//#define STACKVIEW 1
#define INFOCOUNT 5
void forceDirectedLayout(float *,float *,int nodes, float *matrix);

class graph
{


private:


public:
    void allocate(int,int,int,int,int);
    void randAllocate(int);
    void allocateEdgeColor(float,float,float,float);
    void clusterization(char *);
    void convertEdgeMatrixToVerticeList();
    void readGraph(char *);
	void cleanup();
	void readOntology(char *, std::unordered_map<std::string, ontStruct> *);
	void UpdateOntologyInfo(std::string, std::string, std::unordered_map<std::string, ontStruct> *);
	
	
    graph extractGraph(int *);
    float *coords;
    float *coinfo; //dx,dy,dz,radius
    float *color;
    float *edgeMatrix;

    float *coords_d;
    float *coinfo_d; //dx,dy,dz,radius
    float *edgeMatrix_d;


    int *verticeEdgeList;
    int nodes;
    int edges;
    float er,eg,eb,ea;
    float nr,ng,nb,na;
    //std::vector<std::string> nodeVec;
	std::unordered_map<std::string,int> nodeListMap;
	

    int centerx;
    int centery;
    int centerz;
    int width;
    int height;
    char name[256];
    bool displayName;
	std::vector <std::string> *goTerm;

    void addName(char *,int,int,int,int,int);
	graph(int, char *, char *, char *, char *, int, int, int, int, int, std::unordered_map<std::string, ontStruct> *);
	int id;

};

#endif // GRAPH

