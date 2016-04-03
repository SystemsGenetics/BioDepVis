#ifndef _UTIL_H_
#define _UTIL_H_
#include <vector>

char* filetobuf(char *file);
unsigned char * loadBMPRaw(const char * imagepath, unsigned int& outWidth, unsigned int& outHeight, bool flipY = true);

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
#endif