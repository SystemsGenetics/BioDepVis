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

struct nodeSelectedStruct
{
    int nodeSelected;
    int graphSelected;
};

struct ontStruct{
    std::string id;
    std::string name;
    std::string def;
    int index;
    std::vector<nodeSelectedStruct> connectedNodes;
};


class Graph
{



    Graph(int, char *, char *, char *, char *, int, int, int, int, int, std::unordered_map<std::string, ontStruct> *);


};

#endif // GRAPH

