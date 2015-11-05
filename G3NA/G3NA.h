#ifndef _G3NA_H
#define _G3NA_H
#include "graph.h";
#include "alignment.h"
void parser(std::vector <graph*> *, std::vector <Alignment*> *);
void printw(float x, float y, float z, char* format, ...);

struct nodeSelectedStruct
{
	int nodeSelected;
	int graphSelected;
};
#endif
