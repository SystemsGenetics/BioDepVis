//----------------------------------------------
//    Implementation of general utility routines
//----------------------------------------------
//

using namespace std;

#include "util.h"

GLUI *glui;
GLUI *searchglui;
GLUI_EditText *description;
GLUI_TextBox *goDescription;
GLUI_List *gotermList;
GLUI_EditText *searchBox;
GLUI_Button *searchButton;
GLUI_List *selectList;

std::unordered_map<std::string, ontStruct> ontologyDB;

int persp_win;
int WIDTH = 1000;
int HEIGHT = 600;
int MAX_FPS = 60;

float searchRadius = 40;

std::vector <nodeSelectedStruct> selectedVector;
std::vector <nodeSelectedStruct> searchSelectedVector;

GLvoid *font_style = GLUT_BITMAP_HELVETICA_10;
GLvoid *font_style2 = GLUT_BITMAP_TIMES_ROMAN_24;

bool animate = false;
bool cluster = false;
bool showalignment = false;
bool showGrid = false;
bool gpuEnabled = true;
bool searchArea = false;
bool roiMODE = false;
bool backGraphMode = false;

std::vector <nodeSelectedStruct> *SelectedGoPtr;
std::vector <graph*> graphDatabase;
std::vector <Alignment*> alignmentDatabase;

Camera *camera;
unsigned int width_particle=64, height_particle=64;
GLuint textures;

//ROI Pointers
std::vector<float> colorROI;
std::vector<int> verticeEdgeListROI;
std::vector<float> coordsROI;
std::vector<int> alignEdgesROI;
std::vector<int> alignEdgesROI2;
std::vector<int> alignEdgesROI3;

float cx1,cy1,cz1,cx2,cy2,cz2;
float cx12,cy12,cz12,cx22,cy22,cz22;
float cx13,cy13,cz13,cx23,cy23,cz23;

float edgeAlignmentColor[3][4] = {
	{ 189 / MAXCOLOR, 63 / MAXCOLOR, 243 / MAXCOLOR, 0.05f },
	{ 226 / MAXCOLOR, 127 / MAXCOLOR, 202 / MAXCOLOR, 0.05f },
	{ 226 / MAXCOLOR, 127 / MAXCOLOR, 202 / MAXCOLOR, 0.05f }};

int countalignmentdraw= 0;

float xscale=30,yscale=20,zscale=5;
float xscale2=12,yscale2=18,zscale2=5;
float xscale3=30,yscale3=30,zscale3=5;
int nodeSelected2=958,graphSelected2 = 0;
int nodeSelected3=2603,graphSelected3 = 2;

std::vector<unsigned char> loadPNGSimple2(const char* filename, unsigned *width, unsigned *height)
{
	unsigned error;
	std::vector<unsigned char> image;
	GLuint texture;
	unsigned iw, ih;
	error = lodepng::decode(image, iw, ih, filename);
	if (error)
	{
		printf("error %u: %s\n", error, lodepng_error_text(error));
		getchar();
		exit(0);
	}
	*width = iw;
	*height = ih;
	return image;
}


void loadTexture()
{
	std::vector<unsigned char> data_image;
	data_image = loadPNGSimple2("particle.png", &width_particle, &height_particle);

	size_t u2 = 1; while (u2 < width_particle) u2 *= 2;
	size_t v2 = 1; while (v2 < height_particle) v2 *= 2;
	// Ratio for power of two version compared to actual version, to render the non power of two image with proper size.
	double u3 = (double)width_particle / u2;
	double v3 = (double)height_particle / v2;

	// Make power of two version of the image.
	std::vector<unsigned char> image2(u2 * v2 * 4);
	for (size_t y = 0; y < height_particle; y++)
		for (size_t x = 0; x < width_particle; x++)
			for (size_t c = 0; c < 4; c++)
			{
				image2[4 * u2 * y + 4 * x + c] = data_image[4 * width_particle * y + 4 * x + c];
			}

	glGenTextures(1, &textures);
	glBindTexture(GL_TEXTURE_2D, textures);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, 4, u2, v2, 0, GL_RGBA, GL_UNSIGNED_BYTE, &image2[0]);
}


void init()
{
	// set up camera
	// parameters are eye point, aim point, up vector
	camera = new Camera(Vector3d(0, 10, 400), Vector3d(0, 0, 0),
		Vector3d(0, 1, 0));

	// white background for window
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

	loadTexture();
	glEnable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);

	GLfloat light_position[] = { 2.0, -200.0, 0.0, 0.0 };
	GLfloat light_position2[] = { 0.0, 10, -5.0, 0.0 };
	GLfloat light_position3[] = { 0.0, 10, -5.0, 0.0 };
	float specref[] = { 1.0f, 1.0f, 1.0f, 1.0f };

	// Light values and coordinates
	float ambientLight[] = { 0.3f, 0.3f, 0.3f, 1.0f };
	float diffuseLight[] = { 0.7f, 0.7f, 0.7f, 1.0f };
	float specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);

	glLightfv(GL_LIGHT1, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT1, GL_SPECULAR, specular);

	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT1, GL_POSITION, light_position2);
	glLightfv(GL_LIGHT2, GL_POSITION, light_position2);

	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specref);
	glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 128);
	glEnable(GL_COLOR_MATERIAL);

	//glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
}


void drawGraph(graph *g)
{
	glVertexPointer(3, GL_FLOAT, 0, g->coords);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_FALSE);
	glDisable(GL_LIGHTING);
	glLineWidth(0.001f);
	glColor4f(g->er, g->eg, g->eb, g->ea);
	glDrawElements(GL_LINES, g->edges * 2, GL_UNSIGNED_INT, g->verticeEdgeList);
	glDepthMask(GL_TRUE);

 	glColor4f(g->nr, g->ng, g->nb, g->na);

	if (cluster == true)
	{
		glColorPointer(4, GL_FLOAT, 0, g->color);
		glEnableClientState(GL_COLOR_ARRAY);
	}

	glBindTexture(GL_TEXTURE_2D, textures);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

	glPointSize(4.3f);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_FALSE);
	glDrawArrays(GL_POINTS, 0, g->nodes);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_TRUE);

	glDisable(GL_TEXTURE_2D);
	glDisable(GL_POINT_SPRITE);
}


void drawAlignment(Alignment *align)
{
	glDepthMask(GL_FALSE);

	if (showalignment == true)
	{
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, align->vertices);
		glColor4f(edgeAlignmentColor[countalignmentdraw][0],edgeAlignmentColor[countalignmentdraw][1],edgeAlignmentColor[countalignmentdraw][2],edgeAlignmentColor[countalignmentdraw][3] );
		countalignmentdraw = (countalignmentdraw + 1) % alignmentDatabase.size();
		glLineWidth(0.1);
		glDrawArrays(GL_LINES, 0, align->edges);
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	if (animate == false && showalignment == false)
	{
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glColor4f(0.69f, 0.19f, 0.29f, 0.005f);
		glEnable(GL_MAP1_VERTEX_3);
		glLineWidth(0.1);

		for (int k = 0; k < align->edges;  k++)
		{
			float x1 = align->vertices[k * 6 + 0];
			float y1 = align->vertices[k * 6 + 1];
			float z1 = align->vertices[k * 6 + 2];

			float x2 = align->vertices[k * 6 + 3];
			float y2 = align->vertices[k * 6 + 4];
			float z2 = align->vertices[k * 6 + 5];

			float cenerCtrPoint1[3] = { (x1 + x2) / 2.0f, (y1 + y2) / 2.0f, (z1 + z2) / 10.0f - 150.0f };
			float cenerCtrPoint2[3] = { (x1 + x2) / 2.0f, (y1 + y2) / 2.0f, 9 * (z1 + z2) / 10.0f - 150.0f};

			GLfloat ctrlPoints[4][3] = { { align->vertices[k * 6 + 0], align->vertices[k * 6 + 1], align->vertices[k * 6 + 2] }, { cenerCtrPoint1[0], cenerCtrPoint1[1], cenerCtrPoint1[2] }, { cenerCtrPoint2[0], cenerCtrPoint2[1], cenerCtrPoint2[2] }, { align->vertices[k * 6 + 3], align->vertices[k * 6 + 4], align->vertices[k * 6 + 5] } };

			glMap1f(GL_MAP1_VERTEX_3, 0.0, 1.0, 3, 4, &ctrlPoints[0][0]);
			glBegin(GL_LINE_STRIP);
			for (int i = 0; i <= 20; i++)
			{
				GLfloat u = i / (GLfloat)20.0;
				glEvalCoord1f(u);
			}
			glEnd();
		}

		glDisable(GL_MAP1_VERTEX_3);
	}
	glDepthMask(GL_TRUE);
}


void drawAlignmentROI(Alignment *align,int index)
{
	glDepthMask(GL_FALSE);

	if (showalignment == true)
	{
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, align->vertices);
		glColor4f(edgeAlignmentColor[countalignmentdraw][0],edgeAlignmentColor[countalignmentdraw][1],edgeAlignmentColor[countalignmentdraw][2],edgeAlignmentColor[countalignmentdraw][3] );
		countalignmentdraw = (countalignmentdraw + 1) % alignmentDatabase.size();
		glLineWidth(0.1);
		glDrawArrays(GL_LINES, 0, align->edges);
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	if (animate == false && showalignment == false)
	{
		glLoadIdentity();
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glColor4f(0.14,0.24,0.7, 0.15f);

		glEnable(GL_MAP1_VERTEX_3);

		if(index == 0)
		{
			for (int i = 0; i < alignEdgesROI.size();  i=i+2)
			{
				int node1g1 = alignEdgesROI.at(i);
				int node1g2 = alignEdgesROI.at(i+1);
				float x1 = align->g1->coords[node1g1 * 3 + 0];
				float y1 = align->g1->coords[node1g1 * 3 + 1];
				float z1 = align->g1->coords[node1g1 * 3 + 2];

				float x2 = align->g2->coords[node1g2 * 3 + 0];
				float y2 = align->g2->coords[node1g2 * 3 + 1];
				float z2 = align->g2->coords[node1g2 * 3 + 2];

				//if((x1 > cx1 && x1 <  cx2 && y1 > cy1 && y1 <  cy2 && z1 > cz1 && z1 <  cz2 ) && (x2 > cx12 && x2 <  cx22 && y2 > cy12 && y2 <  cy22 && z2 > cz12 && z2 <  cz22 ))
				{

				float cenerCtrPoint1[3] = { (x1 + x2) / 2.0f, (y1 + y2) / 2.0f, (z1 + z2) / 10.0f - 150.0f };
				float cenerCtrPoint2[3] = { (x1 + x2) / 2.0f, (y1 + y2) / 2.0f, 9 * (z1 + z2) / 10.0f - 150.0f};

				GLfloat ctrlPoints[4][3] = { { x1,y1,z1}, { cenerCtrPoint1[0], cenerCtrPoint1[1], cenerCtrPoint1[2] }, { cenerCtrPoint2[0], cenerCtrPoint2[1], cenerCtrPoint2[2] }, { x2,y2,z2 } };
				glMap1f(GL_MAP1_VERTEX_3, 0.0, 1.0, 3, 4, &ctrlPoints[0][0]);

				glLineWidth(0.5);


				glBegin(GL_LINE_STRIP);
				for (int i = 0; i <= 20; i++)
				{
					GLfloat u = i / (GLfloat)20.0;
					glEvalCoord1f(u);
				}
				glEnd();
				}
			}
		}

		if(index == 1)
		{
			for (int i = 0; i < alignEdgesROI2.size();  i=i+2)
			{
				int node1g1 = alignEdgesROI2.at(i);
				int node1g2 = alignEdgesROI2.at(i+1);
				float x1 = align->g1->coords[node1g1 * 3 + 0];
				float y1 = align->g1->coords[node1g1 * 3 + 1];
				float z1 = align->g1->coords[node1g1 * 3 + 2];

				float x2 = align->g2->coords[node1g2 * 3 + 0];
				float y2 = align->g2->coords[node1g2 * 3 + 1];
				float z2 = align->g2->coords[node1g2 * 3 + 2];

				//if((x1 > cx1 && x1 <  cx2 && y1 > cy1 && y1 <  cy2 && z1 > cz1 && z1 <  cz2 ) && (x2 > cx12 && x2 <  cx22 && y2 > cy12 && y2 <  cy22 && z2 > cz12 && z2 <  cz22 ))
				{

				float cenerCtrPoint1[3] = { (x1 + x2) / 2.0f, (y1 + y2) / 2.0f, (z1 + z2) / 10.0f - 150.0f };
				float cenerCtrPoint2[3] = { (x1 + x2) / 2.0f, (y1 + y2) / 2.0f, 9 * (z1 + z2) / 10.0f - 150.0f};

				GLfloat ctrlPoints[4][3] = { { x1,y1,z1}, { cenerCtrPoint1[0], cenerCtrPoint1[1], cenerCtrPoint1[2] }, { cenerCtrPoint2[0], cenerCtrPoint2[1], cenerCtrPoint2[2] }, { x2,y2,z2 } };
				glMap1f(GL_MAP1_VERTEX_3, 0.0, 1.0, 3, 4, &ctrlPoints[0][0]);

				glLineWidth(0.5);

				glBegin(GL_LINE_STRIP);
				for (int i = 0; i <= 20; i++)
				{
					GLfloat u = i / (GLfloat)20.0;
					glEvalCoord1f(u);
				}

				glEnd();
				}
			}
		}

		if(index == 2)
		{
			for (int i = 0; i < alignEdgesROI3.size();  i=i+2)
			{


				int node1g1 = alignEdgesROI3.at(i);
				int node1g2 = alignEdgesROI3.at(i+1);
				float x1 = align->g1->coords[node1g1 * 3 + 0];
				float y1 = align->g1->coords[node1g1 * 3 + 1];
				float z1 = align->g1->coords[node1g1 * 3 + 2];

				float x2 = align->g2->coords[node1g2 * 3 + 0];
				float y2 = align->g2->coords[node1g2 * 3 + 1];
				float z2 = align->g2->coords[node1g2 * 3 + 2];

				//if((x1 > cx1 && x1 <  cx2 && y1 > cy1 && y1 <  cy2 && z1 > cz1 && z1 <  cz2 ) && (x2 > cx12 && x2 <  cx22 && y2 > cy12 && y2 <  cy22 && z2 > cz12 && z2 <  cz22 ))
				{

				float cenerCtrPoint1[3] = { (x1 + x2) / 2.0f, (y1 + y2) / 2.0f, (z1 + z2) / 10.0f - 150.0f };
				float cenerCtrPoint2[3] = { (x1 + x2) / 2.0f, (y1 + y2) / 2.0f, 9 * (z1 + z2) / 10.0f - 150.0f};

				GLfloat ctrlPoints[4][3] = { { x1,y1,z1}, { cenerCtrPoint1[0], cenerCtrPoint1[1], cenerCtrPoint1[2] }, { cenerCtrPoint2[0], cenerCtrPoint2[1], cenerCtrPoint2[2] }, { x2,y2,z2 } };
				glMap1f(GL_MAP1_VERTEX_3, 0.0, 1.0, 3, 4, &ctrlPoints[0][0]);

				glLineWidth(0.5);

				glBegin(GL_LINE_STRIP);
				for (int i = 0; i <= 20; i++)
				{
					GLfloat u = i / (GLfloat)20.0;
					glEvalCoord1f(u);
				}
				glEnd();
				}
			}

			glDisable(GL_MAP1_VERTEX_3);
		}
	}

	glDepthMask(GL_TRUE);
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


int _vscprintf(const char *format, va_list argptr)
{
	return(vsnprintf(0, 0, format, argptr));
}


void printw(float x, float y, float z, char* format, GLvoid *fontStylePrint, ...)
{
	va_list args;	//  Variable argument list
	int len;		//	String length
	int i;			//  Iterator
	char * text;	//	Text

	//  Initialize a variable argument list
	va_start(args, fontStylePrint);

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
	for (i = 0; text[i] != '\0' && i < strlen(text); i++)
		glutBitmapCharacter(fontStylePrint, text[i]);

	//  Free the allocated memory for the string
	free(text);
}


void drawROIBox(int graphSelectedIndex, int nodeSelectedIndex,int xs,int ys,int zs)
{
	//Draw Rect
	glLoadIdentity();
	glBlendFunc (GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glPushMatrix();

	float vx = graphDatabase.at(graphSelectedIndex)->coords[nodeSelectedIndex * 3 + 0];
	float vy = graphDatabase.at(graphSelectedIndex)->coords[nodeSelectedIndex * 3 + 1];
	float vz = graphDatabase.at(graphSelectedIndex)->coords[nodeSelectedIndex * 3 + 2];

	glTranslatef(vx,vy,vz);
	glScalef(xs,ys,zs);
	glColor4f(253.0f/255.0f,212.0f/255.0f,42.0f/255.0,0.8f);
	glLineWidth(2.5);
	glutWireCube(1.0);

	glPopMatrix();
}


void drawROIGraph()
{
	glVertexPointer(3, GL_FLOAT, 0, coordsROI.data());
	glEnableClientState(GL_VERTEX_ARRAY);

 	glColor4f(0.2,0.3,0.5,0.5);

	if (cluster == true)
	{
		glColorPointer(4, GL_FLOAT, 0, colorROI.data());
		glEnableClientState(GL_COLOR_ARRAY);
	}

	glBindTexture(GL_TEXTURE_2D, textures);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

	glPointSize(4.3f);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_FALSE);
	glDrawArrays(GL_POINTS, 0, coordsROI.size()/3);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_TRUE);

	glDisable(GL_TEXTURE_2D);
	glDisable(GL_POINT_SPRITE);
}


void drawGraphROIBack(graph *g)
{
	glColor3f(0.0, 0.0, 0.0);

	if (g->displayName == true)
	{
		printw(g->centerx, g->centery + g->height / 2 - 40, g->centerz, g->name,font_style2);
	}

	glVertexPointer(3, GL_FLOAT, 0, g->coords);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_FALSE);
	glDisable(GL_LIGHTING);
	glLineWidth(0.001f);
	glColor4f(g->er, g->eg, g->eb, 0.10);
	glDrawElements(GL_LINES, g->edges * 2, GL_UNSIGNED_INT, g->verticeEdgeList);
	glDepthMask(GL_TRUE);
	//glEnable(GL_LIGHTING);

 	glColor4f(g->nr, g->ng, g->nb, 0.1);

	if (cluster == true)
	{
		glColorPointer(4, GL_FLOAT, 0, g->color);
		glEnableClientState(GL_COLOR_ARRAY);
	}

	glColor4f(0.0,0.0,0.0,0.1);

	glBindTexture(GL_TEXTURE_2D, textures);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

	glPointSize(4.3f);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_FALSE);

	glDisableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, g->nodes);

	glDisableClientState(GL_VERTEX_ARRAY);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_TRUE);

	glDisable(GL_TEXTURE_2D);
	glDisable(GL_POINT_SPRITE);
}


void PerspDisplay()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// draw the camera created in perspective
	camera->PerspectiveDisplay(WIDTH, HEIGHT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	if(roiMODE == false)
	{
		for (int i = 0; i < alignmentDatabase.size(); i++)
		{
			Alignment *alignT = alignmentDatabase.at(i);
			drawAlignment(alignT);
		}

		for (int i = 0; i < graphDatabase.size(); i++)
		{
			graph *graphT = graphDatabase.at(i);
			drawGraph(graphT);
		}
	}
	else
	{
		if(selectedVector.size() == 0)
			return;

		int nodeSelected = selectedVector.at(0).nodeSelected;
		int graphSelected = selectedVector.at(0).graphSelected;

		drawROIBox(graphSelected,nodeSelected,xscale,yscale,zscale);
		drawROIBox(graphSelected2,nodeSelected2,xscale2,yscale2,zscale2);
		if(alignmentDatabase.size() > 1)
		drawROIBox(graphSelected3,nodeSelected3,xscale3,yscale3,zscale3);

		glLoadIdentity();
		for (int i = 0; i < alignmentDatabase.size(); i++)
		{
			Alignment *alignT = alignmentDatabase.at(i);
			drawAlignmentROI(alignT,i);

			if(backGraphMode)
			{
				for (int i = 0; i < graphDatabase.size(); i++)
				{
					graph *graphT = graphDatabase.at(i);
					drawGraphROIBack(graphT);
				}
			}

			drawROIGraph();
		}
	}

	if (selectedVector.size() > 0)
	{
		for (int i = 0; i < selectedVector.size(); i++)
		{
			int nodeSelected = selectedVector.at(i).nodeSelected;
			int graphSelected = selectedVector.at(i).graphSelected;
			glColor4f(0.0, 0.0, 0.0,1.0);
			float vx = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 0];
			float vy = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 1];
			float vz = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 2];
			glLoadIdentity();
			glPushMatrix();
			glTranslatef(vx,vy,vz);

			glutWireCube(1);
			glColor4f(0.0, 0.0, 0.0,1.0);
			glScalef(0.90, 0.90, 0.90);
			std::string tmp = lookupName(graphSelected, nodeSelected).c_str();
			char name[256]; strncpy(name, tmp.c_str(), tmp.size());
			printw(0 + 0.0, 0 - 1.0, 0 +0.0, name, font_style);
			glPopMatrix();
		}
	}

	if (SelectedGoPtr != NULL)
	{
		for (int i = 0; i < SelectedGoPtr->size(); i++)
		{
			int nodeSelected = SelectedGoPtr->at(i).nodeSelected;
			int graphSelected = SelectedGoPtr->at(i).graphSelected;
			glColor4f(1.0, 0.0, 1.0,0.5);
			float vx = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 0];
			float vy = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 1];
			float vz = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 2];
			glLoadIdentity();
			glPushMatrix();
			glTranslatef(vx, vy, vz);
			glutWireCube(3);
			glPopMatrix();
		}
	}

	if (searchSelectedVector.size() > 0)
	{
		for (int i = 0; i < searchSelectedVector.size(); i++)
		{
			int nodeSelected = searchSelectedVector.at(i).nodeSelected;
			int graphSelected = searchSelectedVector.at(i).graphSelected;

			float vx = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 0];
			float vy = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 1];
			float vz = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 2];
			glLoadIdentity();
			glPushMatrix();
			glColor4f(1.0, 0.0, 0.0,0.02);
			glTranslatef(vx, vy, vz);
			glutWireCube(3);
			glPopMatrix();
		}
	}

	glutSwapBuffers();
}


void timerFunc(int value)
{
	if ( animate ) {
		for (int i = 0; i < graphDatabase.size(); i++)
		{
			graph *g = graphDatabase.at(i);

			if (!gpuEnabled)
				forceDirectedLayout(g->coords, g->coinfo, g->nodes, g->edgeMatrix);
			else
				runForceDirectedGPU(g);
		}

		for (int i = 0; i < alignmentDatabase.size(); i++)
		{
			Alignment *a = alignmentDatabase.at(i);

			if (gpuEnabled){
				runAlignmentForceGPU(a);
			}

			a->update();
		}

		if (gpuEnabled)
		{
			gpuDeviceSync();

			for (int i = 0; i < graphDatabase.size(); i++)
			{
				graph *graphT = graphDatabase.at(i);
				copyForceDirectedGPU(graphT);
			}
		}

		glutSetWindow(persp_win);
		glutPostRedisplay();
	}

	glutTimerFunc(1000 / MAX_FPS, timerFunc, value);
}


void cleanup()
{
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


float PointToLineDistance(const Vector3d &a, const Vector3d &b, const Vector3d &point)
{
	Vector3d line_dir = (b - a).normalize();
	Vector3d point_dir = point - a;
	Vector3d projection = a + (line_dir * (point_dir * line_dir));

	return (projection - point).norm();
}


void AddNodeToROI(int node, graph *tmpg)
{
	float nx = tmpg->coords[node * 3 + 0];
	float ny = tmpg->coords[node * 3 + 1];
	float nz = tmpg->coords[node * 3 + 2];

	int index = coordsROI.size()/3;
	coordsROI.push_back(nx);coordsROI.push_back(ny);coordsROI.push_back(nz);
	colorROI.push_back(tmpg->color[node*4+0]);
	colorROI.push_back(tmpg->color[node*4+1]);
	colorROI.push_back(tmpg->color[node*4+2]);
	colorROI.push_back(tmpg->color[node*4+3]+0.3);
}


bool validROI(int node, graph *tmpg)
{
	float nx = tmpg->coords[node * 3 + 0];
	float ny = tmpg->coords[node * 3 + 1];
	float nz = tmpg->coords[node * 3 + 2];

	if(nx > cx1 && nx <  cx2 && ny > cy1 && ny <  cy2 && nz > cz1 && nz <  cz2)
	{
		return true;
	}
	if(nx > cx12 && nx <  cx22 && ny > cy12 && ny <  cy22 && nz > cz12 && nz <  cz22)
	{
		return true;
	}
	if(alignmentDatabase.size() > 1 && nx > cx13 && nx <  cx23 && ny > cy13 && ny <  cy23 && nz > cz13 && nz <  cz23)
	{
		return true;
	}

	return false;
}

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

	float min = searchArea ? 5.0f : searchRadius;

	selectedVector.clear();
	selectList->delete_all();
	gotermList->delete_all();
	SelectedGoPtr = NULL;

	for (int i = 0; i < graphDatabase.size(); i++)
	{
		for (int j = 0; j < graphDatabase.at(i)->nodes; j++)
		{
			float d = PointToLineDistance(m_start, m_end, Vector3d(graphDatabase.at(i)->coords[j * 3 + 0], graphDatabase.at(i)->coords[j * 3 + 1], graphDatabase.at(i)->coords[j * 3 + 2]));

			if (d < min)
			{
				nodeSelectedStruct tmp;
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
	}

	if (selectedVector.size() > 0)
	{
		//ROI Load
		int nodeSelected = selectedVector.at(0).nodeSelected;
		int graphSelected = selectedVector.at(0).graphSelected;

		if(graphSelected == 0)
			nodeSelected2 = nodeSelected;

		if(graphSelected == 2)
			nodeSelected3 = nodeSelected;

		cx1 = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 0] - (xscale/2) ;
		cy1 = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 1] - (yscale/2) ;
		cz1 = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 2] - (zscale/2) ;

		cx2 = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 0] + (xscale/2) ;
		cy2 = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 1] + (yscale/2) ;
		cz2 = graphDatabase.at(graphSelected)->coords[nodeSelected * 3 + 2] + (zscale/2) ;

		cx12 = graphDatabase.at(graphSelected2)->coords[nodeSelected2 * 3 + 0] - (xscale2/2) ;
		cy12 = graphDatabase.at(graphSelected2)->coords[nodeSelected2 * 3 + 1] - (yscale2/2) ;
		cz12 = graphDatabase.at(graphSelected2)->coords[nodeSelected2 * 3 + 2] - (zscale2/2) ;

		cx22 = graphDatabase.at(graphSelected2)->coords[nodeSelected2 * 3 + 0] + (xscale2/2) ;
		cy22 = graphDatabase.at(graphSelected2)->coords[nodeSelected2 * 3 + 1] + (yscale2/2) ;
		cz22 = graphDatabase.at(graphSelected2)->coords[nodeSelected2 * 3 + 2] + (zscale2/2) ;
	}


	if(alignmentDatabase.size() > 1)
	{
		cx13 = graphDatabase.at(graphSelected3)->coords[nodeSelected3 * 3 + 0] - (xscale3/2) ;
		cy13 = graphDatabase.at(graphSelected3)->coords[nodeSelected3 * 3 + 1] - (yscale3/2) ;
		cz13 = graphDatabase.at(graphSelected3)->coords[nodeSelected3 * 3 + 2] - (zscale3/2) ;

		cx23 = graphDatabase.at(graphSelected3)->coords[nodeSelected3 * 3 + 0] + (xscale3/2) ;
		cy23 = graphDatabase.at(graphSelected3)->coords[nodeSelected3 * 3 + 1] + (yscale3/2) ;
		cz23 = graphDatabase.at(graphSelected3)->coords[nodeSelected3 * 3 + 2] + (zscale3/2) ;
	}

	coordsROI.clear();
	colorROI.clear();
	verticeEdgeListROI.clear();
	alignEdgesROI.clear();
	alignEdgesROI2.clear();
	alignEdgesROI3.clear();

	for (int i = 0; i < alignmentDatabase.size(); i++)
	{
		Alignment *align = alignmentDatabase.at(i);
		for (int k = 0; k < align->edges;  k++)
		{
			int node_g1 = align->g1_vertices.at(k);
			int node_g2 = align->g2_vertices.at(k);

			if(validROI(node_g1,align->g1) && validROI(node_g2,align->g2))
			{
				AddNodeToROI(node_g1,align->g1);
				AddNodeToROI(node_g2,align->g2);
				if(i == 0){
				alignEdgesROI.push_back(node_g1);alignEdgesROI.push_back(node_g2);
				}
				if(i == 1)
				{
				alignEdgesROI2.push_back(node_g1);alignEdgesROI2.push_back(node_g2);
				}
				if(i == 2)
				{
				alignEdgesROI3.push_back(node_g1);alignEdgesROI3.push_back(node_g2);
				}

			}
		}
	}
}
