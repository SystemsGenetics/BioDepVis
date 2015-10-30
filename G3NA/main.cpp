//
// CameraExample.cpp
//
// Example program to show how to use Chris Root's OpenGL Camera Class
// 
// Christopher Root, 2006
// Minor Modifications by Donald House, 2009
// Minor Modifications by Yujie Shu, 2012
//
#include <GL/glew.h>
#include "util.h"
#include "Camera.h"
#ifdef _WIN32
#include "json\json.h"
#else
#include "json/json.h"
#endif
#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL\freeglut.h>
#endif
#include "alignment.h"
#include "graph.h"
#include <cuda_runtime.h>



//  Standard Input\Output C Library
#include <stdio.h>

//  To use functions with variables arguments
#include <stdarg.h>


GLvoid *font_style = GLUT_BITMAP_TIMES_ROMAN_24;

extern "C"
cudaError_t runForceDirectedGPU(graph *g);
extern "C"
cudaError_t runAlignmentForceGPU(Alignment *);
extern "C"
cudaError_t copyForceDirectedGPU(graph *g);
extern "C"
cudaError_t gpuDeviceSync();


//  The number of frames
int frameCount = 0;
//  currentTime - previousTime is the time elapsed
//  between every call of the Idle function
int currentTime = 0, previousTime = 0;


//  Number of frames per second
float fps = 0;

int WIDTH = 1900;
int HEIGHT = 1000;

//Beg of my Dec

bool animate = false;
bool cluster = false;
bool showalignment = false;
bool showGrid = false;
bool gpuEnabled = true;

std::vector <graph*> graphDatabase;
std::vector <Alignment*> alignmentDatabase;

//graph *graph1,*graph2,*graph3, *graph4,*graph5;
//Alignment *alignment_graph,*alignment_betgraph1,*alignment_betgraph2,*alignment_betgraph3;

void printw(float x, float y, float z, char* format, ...);

//End of My Dec    
int persp_win;

Camera *camera;


// draws a simple grid
void makeGrid() {
	glColor3f(0.0, 0.0, 0.0);
	
	glLineWidth(1.0);

	for (float i = -12; i < 12; i++) {
		for (float j = -12; j < 12; j++) {
			glBegin(GL_LINES);
			glVertex3f(i, 0, j);
			glVertex3f(i, 0, j + 1);
			glEnd();
			glBegin(GL_LINES);
			glVertex3f(i, 0, j);
			glVertex3f(i + 1, 0, j);
			glEnd();

			if (j == 11){
				glBegin(GL_LINES);
				glVertex3f(i, 0, j + 1);
				glVertex3f(i + 1, 0, j + 1);
				glEnd();
			}
			if (i == 11){
				glBegin(GL_LINES);
				glVertex3f(i + 1, 0, j);
				glVertex3f(i + 1, 0, j + 1);
				glEnd();
			}
		}
	}

	glLineWidth(2.0);
	glBegin(GL_LINES);
	glVertex3f(-12, 0, 0);
	glVertex3f(12, 0, 0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(0, 0, -12);
	glVertex3f(0, 0, 12);
	glEnd();
	glLineWidth(1.0);
}

char* filereader()
{
	
	int array_size = 65536; // define the size of character array
	char * array = new char[array_size]; // allocating an array of 1kb
	int position = 0; //this will be used incremently to fill characters in the array 

	ifstream fin("testmultiple.json"); //opening an input stream for file test.txt
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
float edgeColor[10][4] = { { 166 / MAXCOLOR, 206, 227, 0.1f },
{ 31 / MAXCOLOR, 120 / MAXCOLOR, 180 / MAXCOLOR, 0.1f },
{ 178 / MAXCOLOR, 223 / MAXCOLOR, 138 / MAXCOLOR, 0.1f },
{ 51 / MAXCOLOR, 160 / MAXCOLOR, 44 / MAXCOLOR, 0.1f },
{ 251 / MAXCOLOR, 154 / MAXCOLOR, 153 / MAXCOLOR, 0.1f },
{ 227 / MAXCOLOR, 26 / MAXCOLOR, 28 / MAXCOLOR, 0.1f },
{ 253 / MAXCOLOR, 191 / MAXCOLOR, 111 / MAXCOLOR, 0.1f },
{ 255 / MAXCOLOR, 127 / MAXCOLOR, 0 / MAXCOLOR, 0.1f },
{ 202 / MAXCOLOR, 178 / MAXCOLOR, 214 / MAXCOLOR, 0.1f },
{ 106 / MAXCOLOR, 61 / MAXCOLOR, 154 / MAXCOLOR, 0.1f } };

void parser()
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
		char *graphname =  new char [graphNameTemp.size() + 1];
		strcpy(graphname, graphNameTemp.c_str()); 
		//graphname[graphNameTemp.size() + 1] = '\0';
		std::string fileloc = graphTemp["fileLocation"].asString(); 
		std::string clusterloc = graphTemp["clusterLocation"].asString();
		int x = graphTemp["x"].asInt();
		int y = graphTemp["y"].asInt();
		int z = graphTemp["z"].asInt();
		int w = graphTemp["w"].asInt();
		int h = graphTemp["h"].asInt();
		printf("<------------>ID : %d\n Name : %s\n Loc: %s\n Cluster : %s\n X: %d\n Y: %d\n Z: %d\n W: %d\n H: %d\n", id, graphNameTemp.c_str(),fileloc.c_str(),clusterloc.c_str(), x, y, z, w, h);
		graph *graphT = new graph(id,graphname, (char *)fileloc.c_str(), (char *)clusterloc.c_str(), x, y, z, w, h);
		graphT->allocateEdgeColor(edgeColor[id-1][0], edgeColor[id-1][1], edgeColor[id][2], edgeColor[id-1][3]);
		graphDatabase.push_back(graphT);
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
			for (int i = 0; i < graphDatabase.size(); i++)
			{
				if (graphDatabase.at(i)->id == leftG)
				{
					indexLeft = i; break;
				}

			}
			for (int i = 0; i < graphDatabase.size(); i++)
			{
				if (graphDatabase.at(i)->id == rightG)
				{
					indexRight = i; break;
				}

			}
			if (indexLeft != -1 && indexRight != -1){
				Alignment *Temp = new Alignment((char *)filelocation.c_str(), (graphDatabase.at(indexLeft)), (graphDatabase.at(indexRight)));
				alignmentDatabase.push_back(Temp);
			}
	}
}




unsigned int m_vertexShader, m_fragmentShader;
char *vertexsource, *fragmentsource;
GLuint shaderprogram;

void init() {
	// set up camera
	// parameters are eye point, aim point, up vector
	cudaError_t cuerr;
	
	parser();
	camera = new Camera(Vector3d(0, 10, 400), Vector3d(0, 0, 0),
		Vector3d(0, 1, 0));

	// grey background for window
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glDepthRange(0.0, 1.0);
	GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat mat_shininess[] = { 50.0 };
	//
	GLfloat light_position[] = { 1.0,1.0,1.0, 0.0 };
	GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);
	/*
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	*/


	/*

	m_vertexShader=  glCreateShader(GL_VERTEX_SHADER);
	m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);


	vertexsource = filetobuf("vertexshader.vert");
	fragmentsource = filetobuf("fragmentshader.frag");

	int IsCompiled_VS, IsCompiled_FS, maxLength, IsLinked;
	char *vertexInfoLog, *fragmentInfoLog, *shaderProgramInfoLog;

	glShaderSource(m_vertexShader, 1, (const GLchar**)&vertexsource, 0);
	glCompileShader(m_vertexShader);
	glGetShaderiv(m_vertexShader, GL_COMPILE_STATUS, &IsCompiled_VS);
	if (IsCompiled_VS == FALSE)
	{
		glGetShaderiv(m_vertexShader, GL_INFO_LOG_LENGTH, &maxLength);

		
		vertexInfoLog = (char *)malloc(maxLength);

		glGetShaderInfoLog(m_vertexShader, maxLength, &maxLength, vertexInfoLog);

		
		printf("Ersror : %s\n", vertexInfoLog);
		free(vertexInfoLog);
		
		getchar();
		exit(0);
	}
	glShaderSource(m_fragmentShader, 1, (const GLchar**)&fragmentsource, 0);
	glCompileShader(m_fragmentShader);
	glGetShaderiv(m_fragmentShader, GL_COMPILE_STATUS, &IsCompiled_FS);
	if (IsCompiled_FS == FALSE)
	{
		glGetShaderiv(m_fragmentShader, GL_INFO_LOG_LENGTH, &maxLength);

		
		fragmentInfoLog = (char *)malloc(maxLength);

		glGetShaderInfoLog(m_fragmentShader, maxLength, &maxLength, fragmentInfoLog);


		printf("Ersror : %s\n", fragmentInfoLog);
		free(fragmentInfoLog);

		getchar();
		exit(0);
	}

	shaderprogram = glCreateProgram();


	glAttachShader(shaderprogram, m_vertexShader);
	glAttachShader(shaderprogram, m_fragmentShader);
	glLinkProgram(shaderprogram);

	glGetProgramiv(shaderprogram, GL_LINK_STATUS, (int *)&IsLinked);
	if (IsLinked == FALSE)
	{
		
		glGetProgramiv(shaderprogram, GL_INFO_LOG_LENGTH, &maxLength);

	
		shaderProgramInfoLog = (char *)malloc(maxLength);

		glGetProgramInfoLog(shaderprogram, maxLength, &maxLength, shaderProgramInfoLog);

		
		printf("Ersror : %s\n", shaderProgramInfoLog);
		free(shaderProgramInfoLog);
		return;
	}
	printf("Vertex Shader ID : %d \n Fragment Shader ID : %d \n", m_vertexShader, m_fragmentShader);
	*/

}

void drawGraph(graph *g)
{

	if (g->displayName == true)
	{
		
		printw(g->centerx, g->centery + g->height / 2, g->centerz, g->name);
	}
	 
	glVertexPointer(3, GL_FLOAT, 0, g->coords);
	glEnableClientState(GL_VERTEX_ARRAY);



	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


	//glDisable(GL_LIGHTING);
	glLineWidth(0.001f);
	glColor4f(g->er, g->eg, g->eb, g->ea);
	glDrawElements(GL_LINES, g->edges * 2, GL_UNSIGNED_INT, g->verticeEdgeList);
	//glEnable(GL_LIGHTING);
	/*float val = (rand()%201)/200.0f * 360.0f;
	float cr,cg,cb;
	HSVtoRGB(&cr,&cg,&cb,val,0.5,0.5);
	glColor3f(cr,cg,cb);*/

	glColor4f(g->nr, g->ng, g->nb, g->na);
	if (cluster == true){
		glColorPointer(4, GL_FLOAT, 0, g->color);
		glEnableClientState(GL_COLOR_ARRAY);
	}

	glUseProgram(shaderprogram);
	glPointSize(4.0f);
	glDrawArrays(GL_POINTS, 0, g->nodes);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisable(GL_BLEND);
	glUseProgram(0);
	glDisableClientState(GL_VERTEX_ARRAY);


}

void runForceDirected(graph *g)
{
	
	if (animate == true){
		if (!gpuEnabled)
			forceDirectedLayout(g->coords, g->coinfo, g->nodes, g->edgeMatrix);
		else
		{
			cudaError_t cuerr = runForceDirectedGPU(g);
			if (cuerr != cudaSuccess)
				cout << "CUDA Error: " << cudaGetErrorString(cuerr) << endl;

		}
	}

}

void runAlignmentLayout(Alignment * a)
{

	if (animate == true)
	{
		if (gpuEnabled){
			cudaError_t cuerr = runAlignmentForceGPU(a);
			if (cuerr != cudaSuccess)
				cout << "CUDA Error: " << cudaGetErrorString(cuerr) << endl;
		}
	}

}


void drawAlignment(Alignment *align)
{

	if (animate == true)
	align->update();
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, align->vertices);
	glColor4f(1.0f, 1.0f, 1.0f, 0.1f);
	glLineWidth(0.1);
	glDrawArrays(GL_LINES, 0, align->edges);
	glDisable(GL_BLEND);

	glDisableClientState(GL_VERTEX_ARRAY);



}



void calculateFPS()
{
	//  Increase frame count
	frameCount++;

	//  Get the number of milliseconds since glutInit called
	//  (or first call to glutGet(GLUT ELAPSED TIME)).
	currentTime = glutGet(GLUT_ELAPSED_TIME);

	//  Calculate time passed
	int timeInterval = currentTime - previousTime;

	if (timeInterval > 1000)
	{
		//  calculate the number of frames per second
		fps = frameCount / (timeInterval / 1000.0f);

		//  Set time
		previousTime = currentTime;

		//  Reset frame count
		frameCount = 0;
	}
}



int _vscprintf(const char *format, va_list argptr)
{
	return(vsnprintf(0, 0, format, argptr));
}

void printw(float x, float y, float z, char* format, ...)
{
	va_list args;	//  Variable argument list
	int len;		//	String length
	int i;			//  Iterator
	char * text;	//	Text

	//  Initialize a variable argument list
	va_start(args, format);

	//  Return the number of characters in the string referenced the list of arguments.
	//  _vscprintf doesn't count terminating '\0' (that's why +1)
	len = _vscprintf(format, args) + 1;

	//  Allocate memory for a string of the specified size
	text = (char *)malloc(len * sizeof(char));

	//  Write formatted output using a pointer to the list of arguments
	vsnprintf(text, len, format, args);
	
	//  End using variable argument list 
	va_end(args);

	//  Specify the raster position for pixel operations.
	glRasterPos3f(x, y, z);

	//  Draw the characters one by one
	for (i = 0; text[i] != '\0'; i++)
		glutBitmapCharacter(font_style, text[i]);

	//  Free the allocated memory for the string
	free(text);
}



void PerspDisplay() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// draw the camera created in perspective
	camera->PerspectiveDisplay(WIDTH, HEIGHT);



	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	if (showGrid)
		makeGrid();

	
	for (int i = 0; i < graphDatabase.size(); i++)
	{
		graph *graphT = graphDatabase.at(i);
		drawGraph(graphT);
		printf("Drawing %s \n ", graphT->name);
	}
		
	for (int i = 0; i < alignmentDatabase.size(); i++)
	{
		Alignment *alignT = alignmentDatabase.at(i);
		drawAlignment(alignT);

	}
	
	
	printf("---\n");

	glutSwapBuffers();
}

void idle()
{
	for (int i = 0; i < graphDatabase.size(); i++)
	{
		graph *graphT = graphDatabase.at(i);

		runForceDirected(graphT);
	}

	for (int i = 0; i < alignmentDatabase.size(); i++)
	{
		Alignment *alignT = alignmentDatabase.at(i);

		runAlignmentLayout(alignT);
	}

	gpuDeviceSync();

	for (int i = 0; i < graphDatabase.size(); i++)
	{
		graph *graphT = graphDatabase.at(i);

		copyForceDirectedGPU(graphT);
	}

	

	glutPostRedisplay();
}

void mouseEventHandler(int button, int state, int x, int y) {
	// let the camera handle some specific mouse events (similar to maya)
	camera->HandleMouseEvent(button, state, x, y);
	glutPostRedisplay();
}
#define SHIFT 15
void motionEventHandler(int x, int y) {
	// let the camera handle some mouse motions if the camera is to be moved
	camera->HandleMouseMotion(x, y);
	glutPostRedisplay();
}

void keyboardEventHandler(unsigned char key, int x, int y) {
	switch (key) {
	case 'r': case 'R':
		// reset the camera to its initial position
		camera->Reset();
		break;
	case 'f': case 'F':
		camera->SetCenterOfFocus(Vector3d(0, 0, 0));
		break;
	case 'g': case 'G':
		gpuEnabled = !gpuEnabled;
		break;

	case ' ':
		animate = !animate;
		break;

	case ',':
		cluster = !cluster;
		break;

	case 'v':
		showalignment = !showalignment;
		break;

		//Camera
	case 'w': case 'W':
		printf("w pressed\n");
		camera->Pos.x += SHIFT;
		break;

	case 's': case 'S':
		printf("w pressed\n");
		camera->Pos.x -= SHIFT;
		break;

	case 'a': case 'A':
		printf("w pressed\n");
		camera->Pos.y += SHIFT;
		break;

	case 'd': case 'D':
		printf("w pressed\n");
		camera->Pos.y -= SHIFT;
		break;

	case 'q': case 'Q':
		printf("w pressed\n");
		camera->Pos.z += SHIFT;
		break;

	case 'e': case 'E':
		printf("w pressed\n");
		camera->Pos.z -= SHIFT;
		break;
		//Aim
	case 'i': case 'I':
		printf("w pressed\n");
		camera->Aim.x += SHIFT;
		break;

	case 'k': case 'K':
		printf("w pressed\n");
		camera->Aim.x -= SHIFT;
		break;

	case 'j': case 'J':
		printf("w pressed\n");
		camera->Aim.y += SHIFT;
		break;

	case 'l': case 'L':
		printf("w pressed\n");
		camera->Aim.y -= SHIFT;
		break;

	case 'u': case 'U':
		printf("w pressed\n");
		camera->Aim.z += SHIFT;
		break;

	case 'o': case 'O':
		printf("w pressed\n");
		camera->Aim.z -= SHIFT;
		break;






	case 27:		// esc
		glDeleteProgram(shaderprogram);
		exit(0);
	}

	glutPostRedisplay();
}

int main(int argc, char *argv[]) {

	// set up opengl window
	glutInit(&argc, argv);
	//glewInit();
	/*
	if (GLEW_ARB_vertex_shader && GLEW_ARB_fragment_shader)
		printf("Ready for GLSL\n");
	else {
		printf("Not totally ready :( \n");
		getchar();
		exit(1);
	}
	*/
	


	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutInitWindowPosition(50, 50);
	persp_win = glutCreateWindow("G3NAV");

	// initialize necessary OpenGL extensions
	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5"))
	{
		fprintf(stderr, "The following required OpenGL extensions missing:\n\tGL_VERSION_2_0\n\tGL_VERSION_1_5\n");
		exit(EXIT_SUCCESS);
	}

	if (!glewIsSupported("GL_ARB_multitexture GL_ARB_vertex_buffer_object GL_EXT_geometry_shader4"))
	{
		fprintf(stderr, "The following required OpenGL extensions missing:\n\tGL_ARB_multitexture\n\tGL_ARB_vertex_buffer_object\n\tGL_EXT_geometry_shader4.\n");
		exit(EXIT_SUCCESS);
	}


	// initialize the camera and such
	init();
	// set up opengl callback functions
	glutDisplayFunc(PerspDisplay);
	glutMouseFunc(mouseEventHandler);
	glutMotionFunc(motionEventHandler);
	glutKeyboardFunc(keyboardEventHandler);
	//glutIdleFunc(PerspDisplay);
	glutIdleFunc(idle);
	glutMainLoop();
	return(0);
}

