#ifndef _G3NA_H
#define _G3NA_H
#include "graph.h"
#include "alignment.h"
#include <unordered_map>
#include <cstdarg>
#include <vector>

void parser(std::vector <graph*> *, std::vector <Alignment*> *, std::unordered_map<std::string, ontStruct> *, const char *f_in);
void readOntFile(std::unordered_map<std::string, ontStruct> *array, const char *ont_input);

#endif
