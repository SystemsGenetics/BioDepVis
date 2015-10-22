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
    std::vector<std::string> nodeVec;

    int centerx;
    int centery;
    int centerz;
    int width;
    int height;
    char *name;
    bool displayName;

    void addName(char *,int,int,int,int,int);
    graph(char *,char *,char *,int,int,int,int,int);

};

#endif // GRAPH

