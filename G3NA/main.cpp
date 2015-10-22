//
// CameraExample.cpp
//
// Example program to show how to use Chris Root's OpenGL Camera Class
// 
// Christopher Root, 2006
// Minor Modifications by Donald House, 2009
// Minor Modifications by Yujie Shu, 2012
//
#include "Camera.h"

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/freeglut.h>
#endif
#include "alignment.h"
#include "graph.h"
//#define SOCIALNETWORK 1
//#define CONSERVENETWORK 1
//#define STACKVIEW 1
#define SINGLEGRAPH 1
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
bool animateAlignment = false;
bool cluster = false;
bool showalignment = false;
graph *graph1,*graph2,*graph3, *graph4,*graph5;
Alignment *alignment_graph,*alignment_betgraph1,*alignment_betgraph2,*alignment_betgraph3;
void printw (float x, float y, float z, char* format, ...);

//End of My Dec    
int persp_win;

Camera *camera;

bool showGrid = false;
bool gpuEnabled = true;

// draws a simple grid
void makeGrid() {
  glColor3f(0.0, 0.0, 0.0);

  glLineWidth(1.0);

  for (float i=-12; i<12; i++) {
    for (float j=-12; j<12; j++) {
      glBegin(GL_LINES);
      glVertex3f(i, 0, j);
      glVertex3f(i, 0, j+1);
      glEnd();
      glBegin(GL_LINES);
      glVertex3f(i, 0, j);
      glVertex3f(i+1, 0, j);
      glEnd();

      if (j == 11){
	glBegin(GL_LINES);
	glVertex3f(i, 0, j+1);
	glVertex3f(i+1, 0, j+1);
	glEnd();
      }
      if (i == 11){
	glBegin(GL_LINES);
	glVertex3f(i+1, 0, j);
	glVertex3f(i+1, 0, j+1);
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

void init() {
  // set up camera
  // parameters are eye point, aim point, up vector
  camera = new Camera(Vector3d(0, 10, 400), Vector3d(0, 0, 0), 
		      Vector3d(0, 1, 0));

  // grey background for window
  glClearColor(0.0,0.0, 0.0,0.0);
  glShadeModel(GL_SMOOTH);
  glDepthRange(0.0, 1.0);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);

  cudaError_t cuerr;

#ifdef CONSERVENETWORK
    //Same Z
    #ifdef STACKVIEW
    graph1 = new graph("Maize","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.tab.cluster",0,0,200,200,200);
    #else
    graph1 = new graph("Maize","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.tab.cluster",200,0,200,200,25);
    #endif
    graph1->allocateEdgeColor(1.0,0.9,0.76,0.2);

    #ifdef STACKVIEW
    graph2 = new graph("Rice","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.tab.cluster",0,0,200,200,-200);
    #else
    graph2 = new graph("Rice","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.tab.cluster",-200,0,200,200,25);
    #endif
    graph2->allocateEdgeColor(0.49,0.80,0.73,0.2);

    #ifdef STACKVIEW
    graph3 = new graph("Conserved Maize","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/grow.M.tab.gna","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.grow.cluster",0,0,200,200,100);
    #else
    graph3 = new graph("Conserved Maize","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/grow.M.tab.gna","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.grow.cluster",200,-300,200,200,25);
    #endif
    graph3->allocateEdgeColor(1.0,0.9,0.76,0.2);

    #ifdef STACKVIEW
    graph4 = new graph("Conserved Rice","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/grow.R.tab.gna","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.grow.cluster",0,0,200,200,-100);
    #else
    graph4 = new graph("Conserved Rice","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/grow.R.tab.gna","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.grow.cluster",-200,-300,200,200,25);
    #endif
    graph4->allocateEdgeColor(0.49,0.80,0.73,0.2);

   alignment_graph = new Alignment("/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/output2.gna",graph3,graph4);
   alignment_betgraph1 = new Alignment("/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M-M.gna",graph1,graph3);
   alignment_betgraph2 = new Alignment("/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R-R.gna",graph2,graph4);
//--------------------------------------------------------------------------------------------------------//
//One Graph
#elif SINGLEGRAPH
    //graph1 = new graph("FACEBOOK","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/facebook.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/facebook.tab.cluster",0,0,200,200,25);
	graph1 = new graph("FACEBOOK", "C:\\Users\\SmaugMe\\Documents\\Visual Studio 2013\\Projects\\G3NA\\Debug\\nvidiadata\\M.tab", "C:\\Users\\SmaugMe\\Documents\\Visual Studio 2013\\Projects\\G3NA\\Debug\\nvidiadata\\M.tab", 0, 0, 200, 200, 25);
       graph1->allocateEdgeColor(0.49,0.80,0.73,0.2);
//--------------------------------------------------------------------------------------------------------//
//Three Graph
#elif THREEGRAPH
    graph1 = new graph("Facebook","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.tab.cluster",-200,200,200,200,25);
       graph1->allocateEdgeColor(1.0,0.9,0.76,0.2);

    graph2 = new graph("Rice","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.tab.cluster",200,200,200,200,25);
       graph2->allocateEdgeColor(0.49,0.80,0.73,0.2);


    graph3 = new graph("MR Alignment","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MR.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MR.tab.cluster",0,-200,200,200,25);
       graph3->allocateEdgeColor(0.80,0.05,0.73,0.2);

 graph4 = new graph("Arabidopsis","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/A.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/A.tab.cluster",300,-200,200,200,25);
       graph4->allocateEdgeColor(0.08,0.55,0.73,0.2);

/* graph5 = new graph("Evolution Seed ","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MRA.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MRA.tab.cluster",100,-500,200,200,25);
       graph5->allocateEdgeColor(0.52,0.25,0.73,0.2);*/


  alignment_betgraph1 = new Alignment("/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MR.tab",graph1,graph3);
  alignment_betgraph2 = new Alignment("/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MR.tab",graph3,graph2);
  alignment_graph = new Alignment("/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MR.tab",graph3,graph4);
 //alignment_betgraph3 = new Alignment("/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MRA.tab",graph5,graph4);




//--------------------------------------------------------------------------------------------------------//
//Two Graph
#elif TWOGRAPH
    graph1 = new graph("Maize","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/M.tab.cluster",-200,200,200,200,25);
       graph1->allocateEdgeColor(1.0,0.9,0.76,0.2);

    graph2 = new graph("Rice","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.tab","/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/R.tab.cluster",200,200,200,200,25);
       graph2->allocateEdgeColor(0.49,0.80,0.73,0.2);


  alignment_betgraph1 = new Alignment("/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MR.tab",graph1,graph2);

 //alignment_betgraph3 = new Alignment("/home/benafsh/Desktop/g3na/nvidiademo2/nvidiadata/MRA.tab",graph5,graph4);
#endif
//

}

void drawGraph(graph *g)
{

     if(g->displayName == true)
	{
	 printw(g->centerx,g->centery+g->height/2,g->centerz,g->name);
	}

	glVertexPointer( 3, GL_FLOAT, 0, g->coords );
	glEnableClientState( GL_VERTEX_ARRAY );



    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);



    glLineWidth(0.001f);
    glColor4f(g->er,g->eg,g->eb,g->ea);

    glDrawElements(GL_LINES,g->edges * 2,GL_UNSIGNED_INT,g->verticeEdgeList);



    

    /*float val = (rand()%201)/200.0f * 360.0f;
    float cr,cg,cb;
    HSVtoRGB(&cr,&cg,&cb,val,0.5,0.5);
    glColor3f(cr,cg,cb);*/
    glColor4f(g->nr,g->ng,g->nb,g->na);
    if(cluster == true){
    glColorPointer( 4, GL_FLOAT, 0, g->color);
    glEnableClientState(GL_COLOR_ARRAY);
    }
    glPointSize(2.0f);
    glDrawArrays(GL_POINTS, 0,g->nodes);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisable(GL_BLEND);

glDisableClientState( GL_VERTEX_ARRAY );
    
    if(animate == true){
     if(!gpuEnabled)
     forceDirectedLayout(g->coords,g->coinfo,g->nodes,g->edgeMatrix);
    else
    		{
       		    cudaError_t cuerr  = runForceDirectedGPU(g);
		    if (cuerr != cudaSuccess)
        	cout << "CUDA Error: " << cudaGetErrorString( cuerr ) << endl;
    	
		}
	}
}



void drawAlignment(Alignment *align,bool animatable = true)
{


    align->update();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, align->vertices);
    glColor4f(1.0f,1.0f,1.0f,0.1f);
    glLineWidth(0.1);
    glDrawArrays(GL_LINES, 0, align->edges);
    glDisable(GL_BLEND);

    glDisableClientState(GL_VERTEX_ARRAY);

    if(animateAlignment == true && animatable == true)
    {
    cudaError_t cuerr  = runAlignmentForceGPU(align);
       // check for errors is always a good practice!
    if (cuerr != cudaSuccess)
        cout << "CUDA Error: " << cudaGetErrorString( cuerr ) << endl;
    }

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
 
    if(timeInterval > 1000)
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

void printw (float x, float y, float z, char* format, ...)
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
	glRasterPos3f (x, y, z);

	//  Draw the characters one by one
    for (i = 0; text[i] != '\0'; i++)
        glutBitmapCharacter(font_style, text[i]);

	//  Free the allocated memory for the string
	free(text);
}


void drawFPS()
{
    //  Load the identity matrix so that FPS string being drawn
    //  won't get animates
	glLoadIdentity ();

	//  Print the FPS to the window
	printw (100, 100, -25, "FPS: %4.2f", fps);
}

void PerspDisplay() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // draw the camera created in perspective
  camera->PerspectiveDisplay(WIDTH, HEIGHT);



  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

//  calculateFPS();
//  drawFPS();

  if (showGrid) 
    makeGrid();
  
  //
  // here is where you would draw your scene!
  //
//  glTranslatef(0, 3.5, 0);
//  glutWireTeapot(5);
  
#ifdef CONSERVENETWORK
    drawGraph(graph1);
    drawGraph(graph2);
    drawGraph(graph3);
    drawGraph(graph4);
    drawAlignment(alignment_graph);
    drawAlignment(alignment_betgraph1);
    drawAlignment(alignment_betgraph2); 
    if(gpuEnabled){
    gpuDeviceSync();
    copyForceDirectedGPU(graph1);copyForceDirectedGPU(graph2);copyForceDirectedGPU(graph3);copyForceDirectedGPU(graph4);
}
#elif SINGLEGRAPH
     drawGraph(graph1);
     if(gpuEnabled){
     gpuDeviceSync();
     copyForceDirectedGPU(graph1);
	}

#elif THREEGRAPH
  drawGraph(graph1);
    drawGraph(graph2);
    drawGraph(graph3);
    drawGraph(graph4);
  //  drawGraph(graph5);
    drawAlignment(alignment_betgraph1,false);
    drawAlignment(alignment_betgraph2,false); 
    drawAlignment(alignment_graph,true);
	//drawAlignment(alignment_betgraph3,false);
     if(gpuEnabled){
     gpuDeviceSync();
     copyForceDirectedGPU(graph1);copyForceDirectedGPU(graph2);copyForceDirectedGPU(graph3);copyForceDirectedGPU(graph4);
	//copyForceDirectedGPU(graph5);
	}

#elif TWOGRAPH
  drawGraph(graph1);
    drawGraph(graph2);
  //  drawGraph(graph5);
	if(showalignment)
    drawAlignment(alignment_betgraph1,true);

	//drawAlignment(alignment_betgraph3,false);
     if(gpuEnabled){
     gpuDeviceSync();
     copyForceDirectedGPU(graph1);copyForceDirectedGPU(graph2);
	//copyForceDirectedGPU(graph5);
	}
#endif

  glutSwapBuffers();
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

    case '/':
   animateAlignment = !animateAlignment;
	break;
//Camera
 case 'w' : case 'W':
        printf("w pressed\n");
        camera->Pos.x += SHIFT;
        break;

    case 's' : case 'S':
        printf("w pressed\n");
        camera->Pos.x -= SHIFT;
        break;

    case 'a' : case 'A':
        printf("w pressed\n");
        camera->Pos.y += SHIFT;
        break;

    case 'd' : case 'D':
        printf("w pressed\n");
        camera->Pos.y -= SHIFT;
        break;

    case 'q' : case 'Q':
        printf("w pressed\n");
        camera->Pos.z += SHIFT;
        break;

    case 'e' : case 'E':
        printf("w pressed\n");
        camera->Pos.z -= SHIFT;
        break;
    //Aim
    case 'i' : case 'I':
        printf("w pressed\n");
        camera->Aim.x += SHIFT;
        break;

    case 'k' : case 'K':
        printf("w pressed\n");
        camera->Aim.x -= SHIFT;
       break;

    case 'j' : case 'J':
        printf("w pressed\n");
        camera->Aim.y += SHIFT;
        break;

    case 'l' : case 'L':
        printf("w pressed\n");
        camera->Aim.y -= SHIFT;
        break;

    case 'u' : case 'U':
        printf("w pressed\n");
        camera->Aim.z += SHIFT;
        break;

    case 'o' : case 'O':
        printf("w pressed\n");
        camera->Aim.z -= SHIFT;
        break;






  case 27:		// esc
    exit(0);
  }

  glutPostRedisplay();
}

int main(int argc, char *argv[]) {

  // set up opengl window
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutInitWindowPosition(50, 50);
  persp_win = glutCreateWindow("Camera Test");

  // initialize the camera and such
  init();

  // set up opengl callback functions
  glutDisplayFunc(PerspDisplay);
  glutMouseFunc(mouseEventHandler);
  glutMotionFunc(motionEventHandler);
  glutKeyboardFunc(keyboardEventHandler);
  glutIdleFunc(PerspDisplay);
  glutMainLoop();
  return(0);
}

