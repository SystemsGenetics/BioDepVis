#include "alignment.h"
#include "graph.h"
#include <stdio.h>
#include <stdlib.h>

void gpuAlignSetup(Alignment *);
void gpuFree(void *);

Alignment::Alignment(char *filename,graph *graph1,graph *graph2)
{
    rows = graph1->nodes;
    cols = graph2->nodes;
    g1 = graph1;
    g2 = graph2;
    edgeAlignMatrix = new float[rows * cols];

    if(edgeAlignMatrix == NULL)
    {
        printf("Matrix cant be created");
        exit(0);

    }
    edges=0;

    readFile(filename);
    gpuAlignSetup(this);

    update();
}

void Alignment::cleanup()
{
	free(edgeAlignMatrix);
	gpuFree(edgeAlignMatrix_d);
}

void Alignment::readFile(char *filename)
{
    FILE *fp2;
    fp2 = fopen(filename,"r");
    char n1[256],n2[256];
 std::string  node1;
 std::string node2;
    while(fscanf(fp2,"%s\t%s\n",n1,n2) == 2 )
    {
       node1 = n1;
       node2 = n2;

		std::unordered_map<std::string, int>::const_iterator found = g1->nodeListMap.find(node1);
		std::unordered_map<std::string, int>::const_iterator found2 =g2->nodeListMap.find(node2);

		if ((found->second) >= 0 && (found2->second) >= 0)
		{
			g1_vertices.push_back((found->second));
			g2_vertices.push_back((found2->second));
			edges++;
		}
		else
		{
			printf("%s <--> %s-error\n", n1, n2);
		}
    }
    std::cout<<"--Total Alignment Edges : "<<edges<<g1_vertices.size()<<","<<g2_vertices.size()<<std::endl;

    int count = 0;
    for(count = 0;count < edges;count++)
    {
        edgeAlignMatrix[g1_vertices.at(count) * cols + g2_vertices.at(count)] = 1.0f;

    }


    count = 0;
   for(int i = 0;i<rows;i++)
       for(int j=0;j<cols;j++)
           if(edgeAlignMatrix[i * cols + j] == 1.0f)
               count++;
       printf("---><--Edge Count Verified : %d vs %d\n",edges, count);


    vertices = new float[edges * 2*3];
}

void Alignment::update()
{
    int k;
    for( k=0;k < edges;k++)
    {
        int ik = g1_vertices.at(k);
        int jk = g1_vertices.at(k);
        vertices[k*6+0] = this->g1->coords[ik * 3 + 0];
        vertices[k*6+1] = this->g1->coords[ik * 3 + 1];
        vertices[k*6+2] = this->g1->coords[ik * 3 + 2];
        vertices[k*6+3] = this->g2->coords[jk * 3 + 0];
        vertices[k*6+4] = this->g2->coords[jk * 3 + 1];
        vertices[k*6+5] = this->g2->coords[jk * 3 + 2];
    }
}
