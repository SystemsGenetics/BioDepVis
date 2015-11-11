//
// CameraExample.cpp
//
// Example program to show how to use Chris Root's OpenGL Camera Class
// 
// Christopher Root, 2006
// Minor Modifications by Donald House, 2009
// Minor Modifications by Yujie Shu, 2012
//

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include "Gl\glui.h"
#else
#include <GL/glew.h>
#endif


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#define GLEW_STATIC
#include "util.h"
#include "Camera.h"

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL\freeglut.h>
#endif
#include "alignment.h"
#include "graph.h"
#include <cuda_runtime.h>
#include "Matrix.h"

#include "G3NA.h"
#ifdef __APPLE__
#include <GL/glui.h>
#else
#include "Gl\glui.h"
#endif

float searchRadius = 40;
GLUI *glui;
GLUI_EditText*description;
GLUI_List *gotermList;

int   wireframe = 0;
int   segments = 8;



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
//int graphSelected = -1, nodeSelected = -1;
std::vector <nodeSelectedStruct> selectedVector;

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
bool searchArea = false;

std::vector <graph*> graphDatabase;
std::vector <Alignment*> alignmentDatabase;

unsigned int m_vertexShader, m_fragmentShader;
char *vertexsource, *fragmentsource;
GLuint shaderprogram;   
int persp_win;
Camera *camera;
unsigned int width_particle = 64, height_particle = 64;
unsigned char *data_particle;
GLuint textures;

void loadTexture()
{

	data_particle = loadBMPRaw("particle.bmp", width_particle, height_particle, false);
	
	
	glGenTextures(1, &textures);
	glBindTexture(GL_TEXTURE_2D, textures);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	


	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_particle, height_particle, 0, GL_BGR, GL_UNSIGNED_BYTE, data_particle);
	//glGenerateMipmap(GL_TEXTURE_2D);
	
	

	delete[] data_particle;
}


int pointSize;
int m_damping;


void init() {
	// set up camera
	// parameters are eye point, aim point, up vector
	cudaError_t cuerr;
	
	parser(&graphDatabase, &alignmentDatabase);
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
	
	loadTexture();
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
	glColor3f(1, 1, 1);
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

	glBindTexture(GL_TEXTURE_2D, textures);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);


	//Orig
	glPointSize(8.0f);
	glDrawArrays(GL_POINTS, 0, g->nodes);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	

	glDisable(GL_TEXTURE_2D);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_BLEND);


	//Orig
	glDisable(GL_BLEND);
}

void Test()
{

	glEnable(GL_BLEND);

	glBindTexture(GL_TEXTURE_2D, textures);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	
	glPointSize(10);
	glBegin(GL_POINTS);
	for (int i = 0; i < 10; i++)
	{

		glVertex3f(0, i * 10, 0);
	}
	glEnd();
	//New
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_BLEND);


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

	for (int i = 0; i < graphDatabase.size(); i++)
	{
		graph *graphT = graphDatabase.at(i);
		drawGraph(graphT);
		//printf("Drawing %s \n ", graphT->name);
	}

	for (int i = 0; i < alignmentDatabase.size(); i++)
	{
		Alignment *alignT = alignmentDatabase.at(i);
		drawAlignment(alignT);

	}
	//Test();
	/*
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, textures);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(0, 0, 200);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(0, 100, 200);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(100,100, 200);
	glTexCoord2f(0.0, 1.0);
	glVertex3f(100, 0, 200);
	glEnd();
	glDisable(GL_TEXTURE_2D);
	*/

	//if (graphSelected != -1 || nodeSelected != -1){
	if (selectedVector.size() > 0){
		for (int i = 0; i < selectedVector.size(); i++){
			int nodeSelected = selectedVector.at(i).nodeSelected;
			int graphSelected = selectedVector.at(i).graphSelected;
			glColor3f(1.0, 1.0, 0.0);
			float vx = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 0];
			float vy = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 1];
			float vz = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 2];
			glLoadIdentity();
			glPushMatrix();
			glTranslatef(vx,vy,vz);
			glutSolidSphere(1, 20, 20);
			glPopMatrix();
		}
	}

	glutSwapBuffers();
}


void idle()
{


	if (glutGetWindow() != persp_win)
		glutSetWindow(persp_win);


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

void cleanup()
{
	printf("Performing Cleanup");
	for (int i = 0; i < graphDatabase.size(); i++)
	{
		graphDatabase.at(i)->cleanup();
		free(graphDatabase.at(i));
	}
	for (int i = 0; i < alignmentDatabase.size(); i++)
	{
		alignmentDatabase.at(i)->cleanup();
		free(alignmentDatabase.at(i));
	}
}

Vector3d ClosestPoint(const Vector3d A, const Vector3d B,
	const Vector3d P, double *t)
{
	Vector3d AB = B - A;
	double ab_square = AB * AB;
	Vector3d AP = P - A;
	double ap_dot_ab = AP * AB;
	// t is a projection param when we project vector AP onto AB 
	*t = ap_dot_ab / ab_square;
	// calculate the closest point 
	Vector3d Q = A + AB * (*t);
	return Q;
}


bool RayTest( const Vector3d start, const Vector3d end, Vector3d center,
	Vector3d *pt, double *t, double epsilon)
{

	*pt = ClosestPoint(start, end, center, t);
	//double len = Distance(*pt, m_pos);
	double len = (*pt - center).norm();
	double m_radius = 5;
	return len < (m_radius + epsilon);
}

bool RayTestPoints(const Vector3d &start, const Vector3d &end,
	unsigned int *id, double *t, double epsilon)
{
	//unsigned int pointID = m_count + 1;
	bool foundCollision = false;
	double minDistToStart = 10000000.0;
	double dst;
	Vector3d pt;

	for (int i = 0; i < graphDatabase.size(); i++)
	{
		for (int j = 0; j < graphDatabase.at(i)->nodes;j++)
		if (RayTest(start, end, graphDatabase.at(i)->coords[j*3 + 0], &pt, t, epsilon))
		{
		
			printf("Valid %d \n---\n", j, graphDatabase.at(i)->nodeListMap["A"]);
		}
	}
	
	return true;
}

float PointToLineDistance(const Vector3d &a, const Vector3d &b, const Vector3d &point){
	Vector3d lineDirection = (b - a).normalize(); Vector3d pointDirection = point - a;
	float t = pointDirection * lineDirection;
	Vector3d projection = a + (lineDirection * t);

	float ShortestDistance = (projection - point).norm();
	return ShortestDistance;
}


std::string lookupName(int graphIndex, int nodeIndex)
{
	for (int i = 0; i < graphDatabase.at(graphIndex)->nodes; i++)
	{
		for (auto it = graphDatabase.at(graphIndex)->nodeListMap.begin(); it != graphDatabase.at(graphIndex)->nodeListMap.end(); ++it)
		{
			if (it->second == nodeIndex)
				return it->first;
		}

	}

}

GLUI_List *selectList;
int listBoxVal;
std::string text = "Hello World!";
int   obj = 0;
int counter = 0;

void GetPickRay(int mouseX, int mouseY)
{
	Vector3d m_start;
	Vector3d m_end;
	double matModelView[16], matProjection[16];
	int viewport[4];
	glLoadIdentity();
	glGetDoublev(GL_MODELVIEW_MATRIX, matModelView);
	glGetDoublev(GL_PROJECTION_MATRIX, matProjection);
	glGetIntegerv(GL_VIEWPORT, viewport);
	double winX = (double)mouseX;
	double winY = viewport[3] - (double)mouseY;
	gluUnProject(winX, winY, 0.0, matModelView, matProjection,
		viewport, &m_start.x, &m_start.y, &m_start.z);
	gluUnProject(winX, winY, 1.0, matModelView, matProjection,
		viewport, &m_end.x, &m_end.y, &m_end.z);
	double t;
	

	float w;
	//int graphSelected = -1;
	//int nodeSelected = -1;
	
	float min = 5.0f;
	if (searchArea != true)
		min = searchRadius;
		//RayTestPoints(m_start, m_end, 0, &t, 0.0001f);
	selectedVector.clear();
	selectList->delete_all();
	for (int i = 0; i < graphDatabase.size(); i++)
	{
		for (int j = 0; j < graphDatabase.at(i)->nodes; j++)
		{
			float d = PointToLineDistance(m_start, m_end, Vector3d(graphDatabase.at(i)->coords[j * 3 + 0], graphDatabase.at(i)->coords[j * 3 + 1], graphDatabase.at(i)->coords[j * 3 + 2]));
			//printf("%d=%f\n", j, d);
			if (d < min)
				{
					nodeSelectedStruct tmp;
				printf("Valid %d = %f\n", j, d);
				tmp.nodeSelected = j;
				tmp.graphSelected = i;
				
				if (searchArea != true)
				{
					min = d;
					selectedVector.clear();
					selectList->delete_all();
					
				}
						selectedVector.push_back(tmp);
						
						char tmpc[256]; 
						std::string nodename = lookupName(tmp.graphSelected, tmp.nodeSelected);
						sprintf(tmpc, "%s",nodename.c_str());
						selectList->add_item(selectedVector.size()-1, tmpc);

				}
			}
		selectList->update_size();
		selectList->update_and_draw_text();
		
		printf("Search Length : %d Less than %f \n----\n",selectedVector.size(),min);
	}
//	printf("graphSelected = %d %d\n", graphSelected, nodeSelected);

}







void mouseEventHandler(int button, int state, int x, int y) {
	// let the camera handle some specific mouse events (similar to maya)
	camera->HandleMouseEvent(button, state, x, y);
	
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP){
		GetPickRay(x, y);
}
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

	case 't': case 'T':
		searchArea = !searchArea;
		break;

	case '+':
		searchRadius=+10;		
		break;
	case '-':
		searchRadius=-10;
		break;

	case 27:		// esc
		//glDeleteProgram(shaderprogram);
		cleanup();
		exit(0);
	}

	glutPostRedisplay();
}




void control_cb(int control)
{

	printf("callback: %d\n", control);
	if (control == 1)
	{
		int id = selectList->get_current_item();
		GLUI_List_Item *item = selectList->get_item_ptr(id);
	
		printf("ID : %d = \n", id);
		description->set_text(item->text);

	}

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
	glutIdleFunc(idle);

	//GLUI
	

	
	glui = GLUI_Master.create_glui("GLUI", GLUI_SUBWINDOW_RIGHT, 0, 0); /* name, flags,x, and y */
	
	new GLUI_Separator(glui);
	new GLUI_StaticText(glui, "Selected Results");
	new GLUI_Separator(glui);
	selectList = new GLUI_List(glui, true, 1, control_cb);
	selectList->set_w(220);
	new GLUI_Separator(glui);
	new GLUI_StaticText(glui, "Description");
	description = new GLUI_EditText(glui, "");
	description->disable();
	description->set_w(220);
	description->set_h(40);
	new GLUI_Separator(glui);
	new GLUI_StaticText(glui, "Go:Term List");
	gotermList = new GLUI_List(glui, true, 2, control_cb);
	gotermList->set_w(220);
	

	glui->set_main_gfx_window(persp_win);

	/* We register the idle callback with GLUI, *not* with GLUT */
	//GLUI_Master.set_glutIdleFunc( myGlutIdle );
	GLUI_Master.set_glutIdleFunc(idle);
	
	
	glutMainLoop();
	return(0);
}

