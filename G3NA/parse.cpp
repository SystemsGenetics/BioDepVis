#ifdef _WIN32
#include "json\json.h"
#else
#include "json/json.h"
#endif

#include "G3NA.h"


//  Standard Input\Output C Library
#include <stdio.h>

//  To use functions with variables arguments
#include <stdarg.h>

#include <iostream>

using namespace std;

char* filereader()
{

	int array_size = 65536; // define the size of character array
	char * array = new char[array_size]; // allocating an array of 1kb
	int position = 0; //this will be used incremently to fill characters in the array 

	ifstream fin("testDual.json"); //opening an input stream for file test.txt
	/*checking whether file could be opened or not. If file does not exist or don't have read permissions, file
	stream could not be opened.*/
	if (fin.is_open())
	{
		//file opened successfully so we are here
		cout << "File Opened successfully!!!. Reading data from file into array" << endl;
		//this loop run until end of file (eof) does not occur
		while (!fin.eof() && position < array_size)
		{
			fin.get(array[position]); //reading one character from file to array
			position++;
		}
		array[position - 1] = '\0'; //placing character array terminating character

		/*cout << "Displaying Array..." << endl << endl;
		//this loop display all the charaters in array till \0
		for (int i = 0; array[i] != '\0'; i++)
		{
		cout << array[i];
		}*/
		return array;
	}
	else //file could not be opened
	{
		cout << "File could not be opened." << endl;
		return NULL;
	}

}

void PrintJSONValue(const Json::Value &val)
{
	if (val.isString()) {
		printf("string(%s)\n", val.asString().c_str());
	}
	else if (val.isBool()) {
		printf("bool(%d)\n", val.asBool());
	}
	else if (val.isInt()) {
		printf("int(%d)\n", val.asInt());
	}
	else if (val.isUInt()) {
		printf("uint(%u)\n", val.asUInt());
	}
	else if (val.isDouble()) {
		printf("double(%f)\n", val.asDouble());
	}
	else
	{
		printf("unknown type=[%d]", val.type());
	}
}

#define MAXCOLOR 255.0f

#define EDGEALPHA 0.2f
float edgeColor[10][4] = { 
	{ 0, 0, 0, EDGEALPHA },
	//{ 166 / MAXCOLOR, 206 / MAXCOLOR, 227 / MAXCOLOR, EDGEALPHA },
	{ 255 / MAXCOLOR, 127 / MAXCOLOR, 0 / MAXCOLOR, EDGEALPHA },
	
{ 106 / MAXCOLOR, 61 / MAXCOLOR, 154 / MAXCOLOR, EDGEALPHA },
{ 31 / MAXCOLOR, 120 / MAXCOLOR, 180 / MAXCOLOR, EDGEALPHA },
{ 178 / MAXCOLOR, 223 / MAXCOLOR, 138 / MAXCOLOR, EDGEALPHA },
{ 51 / MAXCOLOR, 160 / MAXCOLOR, 44 / MAXCOLOR, EDGEALPHA },
{ 251 / MAXCOLOR, 154 / MAXCOLOR, 153 / MAXCOLOR, EDGEALPHA },
{ 227 / MAXCOLOR, 26 / MAXCOLOR, 28 / MAXCOLOR, EDGEALPHA },
{ 253 / MAXCOLOR, 191 / MAXCOLOR, 111 / MAXCOLOR, EDGEALPHA },
{ 255 / MAXCOLOR, 127 / MAXCOLOR, 0 / MAXCOLOR, EDGEALPHA } };

/*
float edgeColor[10][4] = { {179 / MAXCOLOR, 88 / MAXCOLOR, 6 / MAXCOLOR, EDGEALPHA},
{ 84 / MAXCOLOR, 39 / MAXCOLOR, 136 / MAXCOLOR, EDGEALPHA },
{224 / MAXCOLOR, 130 / MAXCOLOR, 20 / MAXCOLOR, EDGEALPHA},
{253 / MAXCOLOR, 184 / MAXCOLOR, 99 / MAXCOLOR, EDGEALPHA},
{254 / MAXCOLOR, 224 / MAXCOLOR, 182 / MAXCOLOR, EDGEALPHA},
{247 / MAXCOLOR, 247 / MAXCOLOR, 247 / MAXCOLOR, EDGEALPHA},
{216 / MAXCOLOR, 218 / MAXCOLOR, 235 / MAXCOLOR, EDGEALPHA},
{178 / MAXCOLOR, 171 / MAXCOLOR, 210 / MAXCOLOR, EDGEALPHA},
{128 / MAXCOLOR, 115 / MAXCOLOR, 172 / MAXCOLOR, EDGEALPHA}
 };*/

void parser(std::vector <graph*> *gd, std::vector <Alignment*> *ad, std::unordered_map<std::string, ontStruct> *ontologyDatabasePtr)
{
	Json::Reader reader;
	Json::Value root;
	char *fileinfo = filereader();
	bool parseStatus = reader.parse(fileinfo, root);
	if (parseStatus == true)
		printf("Parsed Successful");
	else
		printf("Parse Failed");

	Json::Value graphStruct = root["graph"];
	Json::Value alignmentStruct = root["alignment"];

	printf(" {type=[%d], size=%d}", graphStruct.type(), graphStruct.size());


	for (Json::ValueIterator itr = graphStruct.begin(); itr != graphStruct.end(); itr++)
	{
		PrintJSONValue(itr.key());

		Json::Value graphTemp = graphStruct[itr.key().asString().c_str()];
		int id = graphTemp["id"].asInt();
		std::string graphNameTemp = graphTemp["name"].asString();
		char *graphname = new char[graphNameTemp.size() + 1];
		strcpy(graphname, graphNameTemp.c_str());
		//graphname[graphNameTemp.size() + 1] = '\0';
		std::string fileloc = graphTemp["fileLocation"].asString();
		std::string clusterloc = graphTemp["clusterLocation"].asString();
		std::string nodeOntology = graphTemp["Ontology"].asString();
		int x = graphTemp["x"].asInt();
		int y = graphTemp["y"].asInt();
		int z = graphTemp["z"].asInt();
		int w = graphTemp["w"].asInt();
		int h = graphTemp["h"].asInt();
		printf("<------------>ID : %d\n Name : %s\n Loc: %s\n Cluster : %s\n Ont: %s\n X: %d\n Y: %d\n Z: %d\n W: %d\n H: %d\n", id, graphNameTemp.c_str(), fileloc.c_str(), clusterloc.c_str(),nodeOntology.c_str(), x, y, z, w, h);
		graph *graphT = new graph(id, graphname, (char *)fileloc.c_str(), (char *)clusterloc.c_str(), (char *)nodeOntology.c_str(), x, y, z, w, h,ontologyDatabasePtr);
		graphT->allocateEdgeColor(edgeColor[id - 1][0], edgeColor[id - 1][1], edgeColor[id][2], edgeColor[id - 1][3]);
		gd->push_back(graphT);
		printf("Load Complete\n");

	}

	for (Json::ValueIterator itr = alignmentStruct.begin(); itr != alignmentStruct.end(); itr++)
	{
		PrintJSONValue(itr.key());
		Json::Value alignmentTemp = alignmentStruct[itr.key().asString().c_str()];
		int leftG = alignmentTemp["graphID1"].asInt();
		int rightG = alignmentTemp["graphID2"].asInt();
		std::string filelocation = alignmentTemp["filelocation"].asString();
		printf("Perform Alignment on %d %d using %s \n ", leftG, rightG, filelocation.c_str());
		int indexLeft = -1;
		int indexRight = -1;
		for (int i = 0; i < gd->size(); i++)
		{
			if (gd->at(i)->id == leftG)
			{
				indexLeft = i; break;
			}

		}
		for (int i = 0; i < gd->size(); i++)
		{
			if (gd->at(i)->id == rightG)
			{
				indexRight = i; break;
			}

		}
		if (indexLeft != -1 && indexRight != -1){
			Alignment *Temp = new Alignment((char *)filelocation.c_str(), (gd->at(indexLeft)), (gd->at(indexRight)));
			ad->push_back(Temp);
		}
	}
	free(fileinfo);
}

