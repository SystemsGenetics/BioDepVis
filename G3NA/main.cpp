#include "util.h"
#include "events.h"
#include <getopt.h>

typedef enum {
	OPTION_ONTFILE,
	OPTION_JSONFILE,
	OPTION_HELP,
	OPTION_UNKNOWN = '?'
} option_t;


typedef struct {
	char *ont_file;
	char *json_file;
} optarg_t;


void print_usage()
{
	fprintf(stderr,
		"Usage: ./G3NAV [options]\n"
		"\n"
		"Options:\n"
		"  --ont_file   FILE    [REQUIRED] specify the ontology text file\n"
		"  --json_file  FILE    [REQUIRED] specify the json file containing graphs and alignments\n"
		"  --help       list help options\n"
	);
}

//void readOntFile(std::unordered_map<std::string, ontStruct> *);
int main(int argc, char **argv) {

	optarg_t args = {
		NULL,
		NULL
	};

	struct option long_options[] = {
		{ "ont_file", required_argument, 0, OPTION_ONTFILE },
		{ "json_file", required_argument, 0, OPTION_JSONFILE },
		{ "help", no_argument, 0, OPTION_HELP },
		{ 0, 0, 0, 0 }
	};

	int opt;
	while ( (opt = getopt_long_only(argc, argv, "", long_options, NULL)) != -1 ) {
		switch ( opt ) {
		case OPTION_ONTFILE:
			args.ont_file = optarg;
			break;
		case OPTION_JSONFILE:
			args.json_file = optarg;
			break;
		case OPTION_HELP:
			print_usage();
			exit(1);
		case OPTION_UNKNOWN:
			print_usage();
			exit(1);
		}
	}

	// validate arguments
	if ( !args.ont_file || !args.json_file ) {
		print_usage();
		exit(1);
	}

	// read the ontology file
	readOntFile(&ontologyDB, args.ont_file);
	
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

	// create the main window that shows the graphs and alignments
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - WIDTH) / 2, (glutGet(GLUT_SCREEN_HEIGHT) - HEIGHT) / 2);
	persp_win = glutCreateWindow("G3NAV");

	#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
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
	
	#endif

	// initialize the camera and such, pass in the json file to be read
	init(args.json_file);
	
	// set up opengl callback functions
	glutDisplayFunc(PerspDisplay);
	glutMouseFunc(mouseEventHandler);
	glutMotionFunc(motionEventHandler);
	glutKeyboardFunc(keyboardEventHandler);
	glutIdleFunc(idle);

	// MainGLUI
	glui = GLUI_Master.create_glui("GUI", GLUI_SUBWINDOW_RIGHT, 0, 0); /* name, flags,x, and y */
	searchglui = GLUI_Master.create_glui("Search Window", GLUI_SUBWINDOW_TOP, 1800, 0); /* name, flags,x, and y */
	
	new GLUI_Separator(glui);
	new GLUI_StaticText(glui, "Selected Results");
	new GLUI_Separator(glui);
	selectList = new GLUI_List(glui, true, 1, control_cb);
	selectList->set_w(220);
	new GLUI_Separator(glui);
	new GLUI_StaticText(glui, "Description");
	description = new GLUI_EditText(glui, "");
	description->disable();
	description->set_w(420);
	description->set_h(40);
	new GLUI_Separator(glui);
	new GLUI_StaticText(glui, "Go:Term List");
	gotermList = new GLUI_List(glui, true, 2, control_cb);
	gotermList->set_w(220);
	goDescription = new GLUI_TextBox(glui, true);
	goDescription->set_text("GO TERM DESCRIPTION");
	//goDescription->disable();
	goDescription->set_w(420);
	goDescription->set_h(80);


	//Search GLUI
	new GLUI_Separator(searchglui);
	new GLUI_StaticText(searchglui, "Search Term");
	searchBox = new GLUI_EditText(searchglui, "");
	searchButton = new GLUI_Button(searchglui, "Search", 3, control_cb);
	new GLUI_Button(searchglui, "Clear", 4, control_cb);
	searchBox->set_w(420);
	searchBox->set_h(40);
	glui->set_main_gfx_window(persp_win);
	
	
	/* We register the idle callback with GLUI, *not* with GLUT */
	//GLUI_Master.set_glutIdleFunc( myGlutIdle );
	GLUI_Master.set_glutIdleFunc(idle);
	
	
	glutMainLoop();
	return(0);
}

