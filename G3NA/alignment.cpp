#include "alignment.h"
#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

extern "C"
cudaError_t gpuAlignSetup(Alignment *);

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
        //int *index1,*index2;
       // printf("%s %s -->",n1,n2);

        std::vector<std::string>::iterator index1 = std::find(g1->nodeVec.begin(), g1->nodeVec.end(), node1);
        std::vector<std::string>::iterator index2 = std::find(g2->nodeVec.begin(), g2->nodeVec.end(), node2);
       // printf("%d =  %d, %d \n",edges,index1- graph1->nodeVec.begin(),index2-graph2->nodeVec.begin());
        int i1 =index1-(g1->nodeVec.begin());
        int i2 =index2-(g2->nodeVec.begin());
//        edgeAlignMatrix[i1 * rows + i2] = 1.0f;
        if(i1 >= 0 && i2 >= 0)
        {
            g1_vertices.push_back(i1);
            g2_vertices.push_back(i2);
            edges++;
        }
        else
        {
            printf("%s <--> %s-error\n",n1,n2);
        }


    }
    std::cout<<"--Total Alignment Edges : "<<edges<<g1_vertices.size()<<","<<g2_vertices.size()<<std::endl;




    int count = 0;
    for(count = 0;count < edges;count++)
    {
        edgeAlignMatrix[g1_vertices.at(count) * cols + g2_vertices.at(count)] = 1.0f;

    }
    printf("Matrix Alloc Complete\n");


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
    /*
    int k = 0;
    for(int i = 0;i < rows;i++)
    {
        for(int j = 0;j < cols;j++)
        {
            if(edgeAlignMatrix[i * rows + j ]==1.0f)
            {
             //printf("Point : %d\n",k);
            vertices[k*6+0] = this->g1->coords[i * 3 + 0];
            vertices[k*6+1] = this->g1->coords[i * 3 + 1];
            vertices[k*6+2] = this->g1->coords[i * 3 + 2];
            vertices[k*6+3] = this->g2->coords[j * 3 + 0];
            vertices[k*6+4] = this->g2->coords[j * 3 + 1];
            vertices[k*6+5] = this->g2->coords[j * 3 + 2];
            k = k + 1;

            }
        }

    }*/
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
    printf("Edges Moved: %d\n",k);
}




