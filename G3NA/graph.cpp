#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include "color.h"
#include <string>
#include <cuda_runtime.h>
#include <istream>
#include <iostream>
#include <vector>
#include <string>

#define NODEALPHA 1.0f
#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/freeglut.h>
#endif
// This is the 'elder trick of the...' - Tell the compiler this function is defined in other place
extern "C"
cudaError_t gpuSetup(graph *);

graph::graph(int iid, char *graphname, char *filename, char *filenamecluster, char *ontFile, int x, int y, int z, int w, int h, std::unordered_map<std::string, ontStruct> *ontologyDBPtr)
{
	id = iid;
displayName = false;
strncpy(name, graphname, strlen(graphname));
addName(name, x, y, z, w, h);

printf(" Reading %s Filename : %s\n",graphname, filename);
readGraph(filename);
allocate(x,y,w,h,z);
clusterization(filenamecluster);
convertEdgeMatrixToVerticeList();
readOntology(ontFile,ontologyDBPtr);
cudaError_t cuerr;
cuerr=gpuSetup(this);
if (cuerr != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString( cuerr ));
}

void graph::addName(char *graphname,int x,int y,int z,int w,int h)
{
centerx = x;
centery = y;
centerz = z;
width = w;
height = h;
displayName = true;

}

 void graph::readGraph(char* filename)
 {
     er=1.0f;eg=1.0f;eb=1.0f;ea=1.0f;
	 nr = 1.0f; ng = 1.0f; nb = 1.0f, na = NODEALPHA;
     FILE *fp;
     fp = fopen(filename,"r");
     if(fp == NULL)
     {
         printf("File not found :%s\n Exit\n",filename);
         exit(0);
     }
     char n1[256],n2[256];
     //std::string n1,n2;
	 int ctr = 0,ctr2 = 0;
     while(fscanf(fp,"%s\t%s\n",n1,n2) == 2)
     {
        // printf("%s %s\n",n1,n2);
         std::string node1 = n1;
         std::string node2 = n2;
		 

		 //int addMap = -1,addVec = -1;
		 std::unordered_map<std::string, int>::const_iterator found = nodeListMap.find(node1);
		 if (found == nodeListMap.end())
		 {
		//	 std::cout << "Not Found";
			 std::pair<std::string, int> item(node1, ctr);
			 nodeListMap.insert(item);
			 ctr++;
			
		 }
		 else
		 {
		
		 }
		 
		 std::unordered_map<std::string, int>::const_iterator found2 = nodeListMap.find(node2);
		 if (found2 == nodeListMap.end())
		 {
			// std::cout << "Not Found";
			 std::pair<std::string, int> item(node2, ctr);
			 nodeListMap.insert(item);
			 
			 ctr++;
		 }
		 else
		 {
			
		 }
	
		 /*
         if (std::find(nodeVec.begin(), nodeVec.end(), node1) != nodeVec.end())
         {
            //std::cout<<"Found "<<node1<<std::endl;
			 
         }
         else
         {     nodeVec.push_back(node1);   
		 ctr2++;
		 
		 }

         if (std::find(nodeVec.begin(), nodeVec.end(), node2) != nodeVec.end())
         {
             //std::cout<<"Found "<<node2<<std::endl;
         }
         else
         { 
		 nodeVec.push_back(node2);      
		 ctr2++;
		 }*/
		 
     }
/*
     std::cout<<"--Total Nodes : "<<nodeVec.size()<<std::endl;
	 nodes = nodeVec.size();*/
	 std::cout << "--Total Nodes : " << nodeListMap.size() << std::endl;

	 nodes = nodeListMap.size();
	 goTerm = new std::vector<std::string>[nodes];
     edgeMatrix = new float[nodes * nodes];
     edges=0;
     fclose(fp);
     FILE *fp2;
     fp2 = fopen(filename,"r");
     while(fscanf(fp2,"%s\t%s\n",n1,n2) == 2)
     {
         std::string node1 = n1;
         std::string node2 = n2;
         //int *index1,*index2;

        /*std::vector<std::string>::iterator index1 = std::find(nodeVec.begin(), nodeVec.end(), node1);
         std::vector<std::string>::iterator index2 = std::find(nodeVec.begin(), nodeVec.end(), node2);

		 edgeMatrix[(index1 - nodeVec.begin())*nodes + (index2 - nodeVec.begin())] = 1;
		 edgeMatrix[(index2 - nodeVec.begin())*nodes + (index1 - nodeVec.begin())] = 1;*/

		 std::unordered_map<std::string, int>::const_iterator found = nodeListMap.find(node1);
		 std::unordered_map<std::string, int>::const_iterator found2 = nodeListMap.find(node2);
         
		 edgeMatrix[(found->second)*nodes + (found2->second)] = 1;
		 edgeMatrix[(found2->second)*nodes + (found->second)] = 1;

         edges++;
     }
     std::cout<<"--Total Edges : "<<edges<<std::endl;
     fflush(stdout);


 }
 void graph::cleanup()
 {
	 free(edgeMatrix);
	 free(coinfo);
	 free(coords);
	 free(color);

	 cudaFree(coords_d);
	 cudaFree(coinfo_d); //dx,dy,dz,radius
	 cudaFree(edgeMatrix_d);
 }
void graph::allocate(int xc,int yc,int w,int h,int z)
{
    srand(2);
    coords = new float[nodes * 3];
    for(int i = 0; i < nodes;i++)
    {
        coords[i *3 + 0] = rand() % w + -(w/2) + xc;
        coords[i *3 + 1] = rand() % h + -(h/2) + yc;
        coords[i *3 + 2] = z;
    }

    coinfo = new float[nodes * INFOCOUNT];
    for(int i = 0; i < nodes;i++)
    {
    coinfo[i * INFOCOUNT + 0] = 0.0; //dx
    coinfo[i * INFOCOUNT + 1] = 0.0; //dy
    coinfo[i * INFOCOUNT + 2] = 0.0; //dz
    coinfo[i * INFOCOUNT + 3] = 1.0; //radius

    }



}

void graph::clusterization(char *filename)
{
    color = new float[nodes * 4];
    char n1[256],cluster[256];
    //std::string n1,n2;
    FILE *fp2;
    fp2 = fopen(filename,"r");
    while(fscanf(fp2,"%s\t%s\n",n1,cluster) == 2)
    {
        std::string node1 = n1;
        int clusterid = atoi(cluster);
        //int *index1,*index2;

        /*std::vector<std::string>::iterator index1 = std::find(nodeVec.begin(), nodeVec.end(), node1);
		coinfo[(index1 - nodeVec.begin()) * INFOCOUNT + 4] = clusterid;*/

		std::unordered_map<std::string, int>::const_iterator found = nodeListMap.find(node1);
		coinfo[(found->second) * INFOCOUNT + 4] = clusterid;
        

    }
    fflush(stdout);
    fclose(fp2);

    for(int i = 0;i < nodes;i++)
    {
    int cluster = coinfo[i * INFOCOUNT + 4];

    float hval = cluster/60.0f * 360.0f;
    float sval = cluster/60.0f * 1.0f;sval= 0.8f;
    float vval = cluster/60.0f * 1.0f;vval = 0.8f;
    float r,g,b;
    HSVtoRGB(&r,&g,&b,hval,sval,vval);
    if(cluster < 60){
    color[i*4+0]=r;
    color[i*4+1]=g;
    color[i*4+2]=b;
    color[i*4+3]=NODEALPHA;
    }
    else
    {
        color[i*4+0]=1.0f;
        color[i*4+1]=1.0f;
        color[i*4+2]=1.0f;
		color[i * 4 + 3] = NODEALPHA;
    }

    }
}

void graph::allocateEdgeColor(float r, float g, float b, float a )
{
   
    er=r;
    eg=g;
    eb=b;
    ea=a;
}

void graph::randAllocate(int tnodes)
{
    nodes = tnodes;
    srand(1);
    coords = new float[nodes * 3];
    coinfo = new float[nodes * 5];
    for(int i = 0; i < nodes;i++)
    {
        coords[i *3 + 0] = rand() % nodes/2 + (-nodes/2);
        coords[i *3 + 1] = rand() % nodes/2 + (-nodes/2);
        coords[i *3 + 2] = 0;
    }
    edgeMatrix = new float[nodes * nodes];
    edges=0;
    for(int i = 0;i< nodes;i++)
    {
        for(int j =i+1;j < nodes;j++)
        {

            edgeMatrix[ i * nodes + j] = 0;
            edgeMatrix[ j * nodes + i] = 0;
            if(rand()%1000 > 990 && i != j)
            {
                edgeMatrix[ i * nodes + j] = 1;
                edgeMatrix[ j * nodes + i] = 1;
                edges++;
            }
        }
    }


}

void graph::convertEdgeMatrixToVerticeList()
{
    verticeEdgeList = new int[edges * 2];
    int count = 0;
    for(int i = 0;i < nodes;i++)
    {
    for(int j=i+1;j<nodes;j++)
        {
            if(edgeMatrix[i * nodes + j] == 1){
            verticeEdgeList[count * 2 + 0] = i;
            verticeEdgeList[count * 2 + 1] = j;
            //printf("%d - %d\n",i,j);
            count++;
            }


        }

    }
    printf("Count = %d vs %d\n",count  ,edges);
    fflush(stdout);
}

void graph::UpdateOntologyInfo(std::string name, std::string golistString, std::unordered_map<std::string, ontStruct> *ontologyDatabasePtr)
{
	std::vector<std::string> goList;
	std::stringstream ss(golistString);

	std::string term;

	while (std::getline(ss, term, ',')) {
		//std::cout << term << '\n';

		goList.push_back(term);
	}

	/*while (ss >> term)
	{
		goList.push_back(term);

		if (ss.peek() == ',')
			ss.ignore();
	}*/

	
	
	std::unordered_map<std::string, int>::const_iterator found = nodeListMap.find(name);

	if (found == nodeListMap.end())
	{
		return;
	}
	
	for (int i = 0; i < goList.size(); i++)
		if (std::find(goTerm[found->second].begin(), goTerm[found->second].end(), goList.at(i)) != goTerm[found->second].end()) {
			/* v contains x */
			
		}
		else {
			/* v does not contain x */
			nodeSelectedStruct tmp;
			tmp.graphSelected = id-1; tmp.nodeSelected = found->second;
			goTerm[found->second].push_back(goList.at(i));
			std::unordered_map<std::string, ontStruct>::const_iterator foundOnt = ontologyDatabasePtr->find(goList.at(i));
			if (foundOnt != ontologyDatabasePtr->end())
			ontologyDatabasePtr->at(goList.at(i)).connectedNodes.push_back(tmp);
			
		}
		


}

void graph::readOntology(char *filename, std::unordered_map<std::string, ontStruct> *ontologyDBPtr)
{
	typedef std::vector<std::vector<std::string> > Rows;
	Rows rows;
	std::ifstream input(filename);
	char const row_delim = '\n';
	char const field_delim = '\t';
	for (std::string row; getline(input, row, row_delim);) {
		rows.push_back(Rows::value_type());
		std::istringstream ss(row);
		int index = 0;
		std::string  name;
		std::string goterm;
		for (std::string field; getline(ss, field, field_delim);) {
			rows.back().push_back(field);

			if (index == 1)
				name = field;
			if (index == 9){
				goterm = field;
				std::unordered_map<std::string, int>::const_iterator found = nodeListMap.find(name);
				if (found != nodeListMap.end() && goterm!= "")
					UpdateOntologyInfo(name, goterm,ontologyDBPtr);
				break;
			}

			index++;
			
		}

	}
}



/*std::unordered_map<std::string, int>::const_iterator found = nodeListMap.find(name);

if (found != nodeListMap.end())
{
	UpdateOntologyInfo(name, goterm);
}*/


