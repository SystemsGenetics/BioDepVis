#ifndef _UTIL_H_
#define _UTIL_H_
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <GL/glui.h>
#include <cuda_runtime.h>

#include "alignment.h"
#include "graph.h"
#include "Matrix.h"
#include "G3NA.h"
#include "lodepng.h"
#include "Camera.h"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/glew.h>
#include "Gl\glui.h"
#else
#include <GL/glui.h>
//#include <GL/glew.h>
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/freeglut.h>
#endif

#ifdef __APPLE__
#include <GL/glui.h>
#elif (WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include "Gl/glui.h"
#endif

#define GLEW_STATIC
#define MAXCOLOR 255.0f
#define SHIFT 15

// global variables for GLUT objects
extern GLUI *glui;
extern GLUI *searchglui;
extern GLUI_EditText *description;
extern GLUI_TextBox *goDescription;
extern GLUI_List *gotermList;
extern GLUI_EditText *searchBox;
extern GLUI_Button *searchButton;
extern GLUI_List *selectList;
extern Camera *camera;

// global variables that are modified during events
extern bool animate;
extern bool cluster;
extern bool showalignment;
extern bool showGrid;
extern bool gpuEnabled;
extern bool searchArea;
extern bool roiMODE;
extern bool backGraphMode;
extern float yscale, yscale2;
extern float xscale, xscale2;
extern float searchRadius;
extern std::vector <nodeSelectedStruct> selectedVector;
extern std::vector <nodeSelectedStruct> searchSelectedVector;
extern std::vector <nodeSelectedStruct> *SelectedGoPtr;
extern std::vector <graph*> graphDatabase;

// ontolgoy database used in several files
extern std::unordered_map<std::string, ontStruct> ontologyDB;

extern int persp_win;

extern int WIDTH;
extern int HEIGHT;

// void printw(float x, float y, float z, char* format, GLvoid *, ...);
extern "C"
cudaError_t runForceDirectedGPU(graph *g);
extern "C"
cudaError_t runAlignmentForceGPU(Alignment *);
extern "C"
cudaError_t copyForceDirectedGPU(graph *g);
extern "C"
cudaError_t gpuDeviceSync();


// function prototypes
char* filetobuf(char *file);
unsigned char * loadBMPRaw(const char * imagepath, unsigned int& outWidth, unsigned int& outHeight, bool flipY = true);
std::vector<unsigned char> loadPNGSimple2(const char* filename, unsigned *width, unsigned *height);
void loadTexture();
void init(const char * f_in);
void drawGraph(graph *g);
void Test();
void runForceDirected(graph *g);
void runAlignmentLayout(Alignment * a);
void drawAlignment(Alignment *align);
void drawAlignmentROI(Alignment *align,int index);
std::string lookupName(int graphIndex, int nodeIndex);
int _vscprintf(const char *format, va_list argptr);
void printw(float x, float y, float z, char* format, GLvoid *fontStylePrint, ...);
void drawROIBox(int graphSelectedIndex, int nodeSelectedIndex,int xs,int ys,int zs);
void drawROIGraph();
void drawGraphROIBack(graph *g);
void drawPoints();
void PerspDisplay();
void idle();
void cleanup();
Vector3d ClosestPoint(const Vector3d A, const Vector3d B, const Vector3d P, double *t);
bool RayTest( const Vector3d start, const Vector3d end, Vector3d center, Vector3d *pt, double *t, double epsilon);
bool RayTestPoints(const Vector3d &start, const Vector3d &end, unsigned int *id, double *t, double epsilon);
float PointToLineDistance(const Vector3d &a, const Vector3d &b, const Vector3d &point);
void AddNodeToROI(int node, graph *tmpg);
bool validROI(int node, graph *tmpg);
void GetPickRay(int mouseX, int mouseY);

#endif
