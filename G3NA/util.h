#ifndef _UTIL_H_
#define _UTIL_H_
char* filetobuf(char *file);
unsigned char * loadBMPRaw(const char * imagepath, unsigned int& outWidth, unsigned int& outHeight, bool flipY = true);
#endif