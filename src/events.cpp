#include "util.h"
#include "events.h"

void mouseEventHandler(int button, int state, int x, int y)
{
	// let the camera handle some specific mouse events (similar to maya)
	camera->HandleMouseEvent(button, state, x, y);

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		GetPickRay(x, y);
	}
	glutPostRedisplay();
}


void motionEventHandler(int x, int y)
{
	// let the camera handle some mouse motions if the camera is to be moved
	camera->HandleMouseMotion(x, y);
	glutPostRedisplay();
}


void keyboardEventHandler(unsigned char key, int x, int y)
{
	switch (key) {
	case 'r': case 'R':
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
	case 'v': case 'V':
		showalignment = !showalignment;
		break;
	case 'w': case 'W':
		camera->Pos.x += SHIFT;
		break;
	case 's': case 'S':
		camera->Pos.x -= SHIFT;
		break;
	case 'a': case 'A':
		camera->Pos.y += SHIFT;
		break;
	case 'd': case 'D':
		camera->Pos.y -= SHIFT;
		break;
	case 'q': case 'Q':
		camera->Pos.z += SHIFT;
		break;
	case 'e': case 'E':
		camera->Pos.z -= SHIFT;
		break;
	case 'j': case 'J':
		camera->Aim.x -= SHIFT;
		break;
	case 'l': case 'L':
		camera->Aim.x += SHIFT;
		break;
	case 'i': case 'I':
		camera->Aim.y += SHIFT;
		break;
	case 'k': case 'K':
		camera->Aim.y -= SHIFT;
		break;
	case 't': case 'T':
		searchArea = !searchArea;
		break;
	case 'p':
		yscale+=2;
		break;
	case '[':
		xscale+=2;
		break;
	case 'P':
		yscale-=2;
		break;
	case '{':
		xscale-=2;
		break;
	case '.':
		yscale2+=2;
		break;
	case '/':
		xscale2+=2;
		break;
	case '>':
		yscale2-=2;
		break;
	case '?':
		xscale2-=2;
		break;
	case '+':
		searchRadius=+10;
		break;
	case '-':
		searchRadius=-10;
		break;
	case 'x': case 'X':
		roiMODE=!roiMODE;
		break;
	case 'z': case 'Z':
		backGraphMode=!backGraphMode;
		break;
	case 27:
		cleanup();
		exit(0);
	}

	glutPostRedisplay();
}

std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string> &elems)
{
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

std::vector<std::string> split(const std::string &s, char delim)
{
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}


void control_cb(int control)
{
	//Genes
	if (control == 1)
	{
		int id = selectList->get_current_item();
		GLUI_List_Item *item = selectList->get_item_ptr(id);

		printf("ID : %d = \n", id);
		description->set_text(item->text.c_str());
		gotermList->delete_all();

		std::vector <std::string> goList = graphDatabase.at(selectedVector[id].graphSelected)->goTerm[selectedVector[id].nodeSelected];
		for (int i = 0; i < goList.size();i++)
			gotermList->add_item(i, goList.at(i).c_str());
	}

	//Go Terms
	if (control == 2)
	{
		int id = gotermList->get_current_item();
		GLUI_List_Item *item = gotermList->get_item_ptr(id);
		std::string goname = item->text;
		std::string setText = "Name : " + ontologyDB.at(goname).name + "\n" + ontologyDB.at(goname).def.c_str();
		goDescription->set_text(setText.c_str());
		SelectedGoPtr = &(ontologyDB.at(goname).connectedNodes);
	}

	//Search Box
	if (control == 3)
	{
		selectList->delete_all();
		gotermList->delete_all();
		description->set_text("");
		selectList->redraw();
		gotermList->redraw();
		std::string searchString = searchBox->get_text();
		std::vector<std::string> searchList = split(searchString, '+');

		searchSelectedVector.clear();

		for (auto local_it = ontologyDB.begin(); local_it != ontologyDB.end(); ++local_it)
		{
			bool foundStatus = true;
			for (int i = 0; i < searchList.size(); i++)
			{
				std::size_t found = local_it->second.def.find(searchList.at(i));
				if (found == std::string::npos)
				{
					foundStatus = false;
					break;
				}

			}

			if (foundStatus == true)
			{
				for (auto nodeIt = local_it->second.connectedNodes.begin(); nodeIt != local_it->second.connectedNodes.end(); ++nodeIt)
				{
					nodeSelectedStruct tmp;
					tmp.graphSelected = nodeIt->graphSelected; tmp.nodeSelected = nodeIt->nodeSelected;
					searchSelectedVector.push_back(tmp);
				}
			}
		}
	}

	if (control == 4)
	{
		searchSelectedVector.clear();
	}
}
