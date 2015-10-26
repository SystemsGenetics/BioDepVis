#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda.h>
#include "graph.h"
#include "alignment.h"
#define MAX_DISPLACEMENT_SQUARED 10.0f
// Kernel that executes on the CUDA device


extern "C"
cudaError_t gpuSetup(graph *g)
{
  int graphsize = g->nodes;
  printf("Graph Size :%d for GPU\n",graphsize);
  //Mallocing Edge Space
  printf("Malloc Edge Matrix\n");
  cudaMalloc((void **)&(g->edgeMatrix_d),graphsize * graphsize * sizeof(float));
  printf("Malloc Node Position\n");
  cudaMalloc((void**)&(g->coords_d), graphsize * sizeof(float) * 3 );
  printf("Malloc Node Prop\n");
  cudaMalloc((void**)&(g->coinfo_d),graphsize * sizeof(float) * INFOCOUNT);
  printf("Mem Cpy Node Position\n");
  cudaMemcpy(g->coords_d,g->coords,graphsize * sizeof(float) * 3, cudaMemcpyHostToDevice);
  printf("Mem Cpy Node Prop\n");
  cudaMemcpy(g->coinfo_d, g->coinfo,graphsize * sizeof(float) * INFOCOUNT,cudaMemcpyHostToDevice);
  printf("Mem Cpy Edge\n");
  cudaMemcpy(g->edgeMatrix_d,g->edgeMatrix,sizeof(float) * graphsize * graphsize, cudaMemcpyHostToDevice);
  printf("Setup Complete \n");
  return cudaGetLastError();
}

extern "C"
cudaError_t gpuAlignSetup(Alignment *align)
{
  //Mallocing Edge Space
  printf("Malloc Align Edge Matrix\n");
  cudaMalloc((void **)&(align->edgeAlignMatrix_d),align->rows * align->cols * sizeof(float));
  printf("Mem Cpy Edge Align\n");
  cudaMemcpy(align->edgeAlignMatrix_d,align->edgeAlignMatrix,sizeof(float) * align->rows * align->cols, cudaMemcpyHostToDevice);
  printf("Setup Complete \n");
  return cudaGetLastError();
}

extern "C"
cudaError_t gpuFree(graph *g)
{
    printf("Free Initiated\n");
    cudaFree(g->coords_d);
cudaFree(g->coinfo_d);
cudaFree(g->edgeMatrix_d);
printf("Free Complete\n");
return cudaGetLastError();
}


__global__ void testforceDirectedKernel(int graphsize,float *nodePosition, float *nodeProperty,float *matrixEdge)
{
int id = threadIdx.x + blockIdx.x *  blockDim.x;
if(id < graphsize){
//	printf("%d => %2.3f %2.3f %2.3f\n",id,nodePosition[id].x, nodePosition[id].y, nodePosition[id].z);
    nodePosition[id * 3 + 1] += 1.1f;
    }
}


__global__ void forceDirectedKernel3d(int graphsize,float *nodePosition, float *nodeProperty, float *matrixEdge)
{
int id = threadIdx.x + blockIdx.x *  blockDim.x;
if(id < graphsize){
    //printf("%d => %2.3f %2.3f %2.3f\n",id,nodePosition[id].x, nodePosition[id].y, nodePosition[id].z);
    float K_r = 25.0f; //2
    float K_s = 15.0f; //1
    float L = 1.2f;
    float delta_t = 0.004f;
    int i,j = 0;
    i = id;
    {
        for(j = 0;j < graphsize;j++)
        {
            float dx = nodePosition[j*3+0] - nodePosition[i*3+0];
            float dy = nodePosition[j*3+1] - nodePosition[i*3+1];
            float dz = nodePosition[j*3+2] - nodePosition[i*3+2];
            if(dx != 0 && dy!= 0 && dz!=0  && i!=j )
            {
                float distance  = sqrt(dx * dx + dy * dy + dz * dz);
                float force =0.0f,force2=0.0f;


                if(matrixEdge[i * graphsize + j] == 0)
                force = K_r / (distance * distance);
                if(matrixEdge[i * graphsize + j] == 1)
                force2 =  K_s * (distance - L);
            nodeProperty[i*INFOCOUNT+0] = nodeProperty[i*INFOCOUNT+0] - ((force * dx)/distance) +((force2*dx)/distance) ;
            nodeProperty[i*INFOCOUNT+1] = nodeProperty[i*INFOCOUNT+1] - ((force * dy)/distance) +((force2*dy)/distance) ;
            nodeProperty[i*INFOCOUNT+2] = nodeProperty[i*INFOCOUNT+2] - ((force * dz)/distance) +((force2*dz)/distance) ;
            nodeProperty[j*INFOCOUNT+0] = nodeProperty[j*INFOCOUNT+0] + ((force * dx)/distance) -((force2*dx)/distance) ;
            nodeProperty[j*INFOCOUNT+1] = nodeProperty[j*INFOCOUNT+1] + ((force * dy)/distance) -((force2*dy)/distance) ;
            nodeProperty[j*INFOCOUNT+2] = nodeProperty[j*INFOCOUNT+2] + ((force * dz)/distance) -((force2*dz)/distance) ;
            }

        }

    __syncthreads();
        float d_x = delta_t * nodeProperty[i * INFOCOUNT + 0];
        float d_y = delta_t * nodeProperty[i * INFOCOUNT + 1];
        float d_z = delta_t * nodeProperty[i * INFOCOUNT + 2];
        float displacementSquared = d_x*d_x + d_y*d_y + d_z*d_z;
        if ( displacementSquared > MAX_DISPLACEMENT_SQUARED ){
            float s = sqrt( MAX_DISPLACEMENT_SQUARED  / displacementSquared );
            d_x = d_x * s;
            d_y = d_y * s;
        d_z = d_z * s;
        }
        nodePosition[i*3+2] += d_z;
        nodePosition[i*3+1] += d_y;
        nodePosition[i*3+0] += d_x;
        nodeProperty[i*INFOCOUNT+0] *= .06f;
        nodeProperty[i*INFOCOUNT+1] *= .06f;
        nodeProperty[i*INFOCOUNT+2] *= .06f;
    }
  }
}

__global__ void forceDirectedKernel2d(int graphsize,float *nodePosition, float *nodeProperty, float *matrixEdge)
{
int id = threadIdx.x + blockIdx.x *  blockDim.x;
if(id < graphsize){
    //printf("%d => %2.3f %2.3f %2.3f\n",id,nodePosition[id].x, nodePosition[id].y, nodePosition[id].z);
    float K_r = 25.0f; //2
    float K_s = 15.0f; //1
    float L = 1.2f;
    float delta_t = 0.004f;
    int i,j = 0;
    i = id;
    {
        for(j = 0;j < graphsize;j++)
        {
            float dx = nodePosition[j*3+0] - nodePosition[i*3+0];
            float dy = nodePosition[j*3+1] - nodePosition[i*3+1];
            if(dx != 0 && dy!= 0 && i!=j )
            {
                float distance  = sqrt(dx * dx + dy * dy);
                float force =0.0f,force2=0.0f;


                if(matrixEdge[i * graphsize + j] == 0)
                force = K_r / (distance * distance);
                if(matrixEdge[i * graphsize + j] == 1)
                force2 =  K_s * (distance - L);
            nodeProperty[i*INFOCOUNT+0] = nodeProperty[i*INFOCOUNT+0] - ((force * dx)/distance) +((force2*dx)/distance) ;
            nodeProperty[i*INFOCOUNT+1] = nodeProperty[i*INFOCOUNT+1] - ((force * dy)/distance) +((force2*dy)/distance) ;
            nodeProperty[j*INFOCOUNT+0] = nodeProperty[j*INFOCOUNT+0] + ((force * dx)/distance) -((force2*dx)/distance) ;
            nodeProperty[j*INFOCOUNT+1] = nodeProperty[j*INFOCOUNT+1] + ((force * dy)/distance) -((force2*dy)/distance) ;
            }

        }

    __syncthreads();
        float d_x = delta_t * nodeProperty[i * INFOCOUNT + 0];
        float d_y = delta_t * nodeProperty[i * INFOCOUNT + 1];
        float displacementSquared = d_x*d_x + d_y*d_y;
        if ( displacementSquared > MAX_DISPLACEMENT_SQUARED ){
            float s = sqrt( MAX_DISPLACEMENT_SQUARED  / displacementSquared );
            d_x = d_x * s;
            d_y = d_y * s;
        }
        nodePosition[i*3+2] += 0.0f;
        nodePosition[i*3+1] += d_y;
        nodePosition[i*3+0] += d_x;
        nodeProperty[i*INFOCOUNT+0] *= .06f;
        nodeProperty[i*INFOCOUNT+1] *= .06f;
    }
  }
}

extern "C"
cudaError_t runForceDirectedGPU(graph *g)
{
printf("running kernel\n");
//testforceDirectedKernel<<<(int)(g->nodes/256)+1,256>>>(g->nodes,g->coords_d,g->coinfo_d,g->edgeMatrix_d);
forceDirectedKernel2d<<<(int)(g->nodes/256)+1,256>>>(g->nodes,g->coords_d,g->coinfo_d,g->edgeMatrix_d);
//cudaDeviceSynchronize();
return cudaGetLastError();
}

extern "C"
cudaError_t copyForceDirectedGPU(graph *g)
{
    cudaMemcpy(g->coords,g->coords_d,g->nodes * sizeof(float)*3, cudaMemcpyDeviceToHost);
    return cudaGetLastError();
}

extern "C"
cudaError_t gpuDeviceSync()
{
    cudaDeviceSynchronize();
	return cudaGetLastError();
}


__global__ void forceAlignKernel2d(int rows,int cols ,float *graph1Pos, float *graph2Pos,float *matrixEdge)
{

int id = threadIdx.x + blockIdx.x *  blockDim.x;
if(id < rows){
//	printf("%d => %2.3f %2.3f %2.3f\n",id,nodePosition[id].x, nodePosition[id].y, nodePosition[id].z);
    for(int j = 0;j < cols;j++)
    if(matrixEdge[id * cols + j] == 1)
    {
     if(graph1Pos[id*3+1] + 2.0f < -400.0f)
    {
     graph1Pos[id*3+0] = 0.0f;
    graph1Pos[id*3+1] += 5.0f;
    }
    if(graph2Pos[j*3+1] + 2.0f <-400.0f)

    {
    graph2Pos[j*3+0] = 0.0f;
    graph2Pos[j*3+1] += 5.0f;
    }
    }
}

}

#define radius 2
__global__ void forceAlignStackKernel2d(int rows,int cols ,float *nodePosition1, float *nodePosition2,float *nodeProperty1,float *nodeProperty2,float *matrixEdge)
{

int id = threadIdx.x + blockIdx.x *  blockDim.x;
if(id < rows){
//	printf("%d => %2.3f %2.3f %2.3f\n",id,nodePosition[id].x, nodePosition[id].y, nodePosition[id].z);
    for(int j = 0;j < cols;j++)
    {
    float xdiff=0.0,ydiff =0.0;
    if(matrixEdge[id * cols + j] == 1.00f && fabs(nodePosition2[j * 3 + 0 ] - nodePosition1[id * 3 + 0 ]) < 4.0f )
    {
         xdiff += (nodePosition2[j * 3 + 0 ] - nodePosition1[id * 3 + 0 ]);
        ydiff += (nodePosition2[j * 3 + 1 ] - nodePosition1[id * 3 + 1 ]);

        //nodePosition1[id * 3 + 2 ] = 45.0f;
        //nodePosition2[j * 3 + 2 ] = -45.0f;
    }
    __syncthreads();
    nodePosition1[id * 3 + 0 ] += xdiff*.9;
    nodePosition1[id * 3 + 1 ] += ydiff*.9;

    }
}



}
extern "C"
cudaError_t runAlignmentForceGPU(Alignment *algn)
{
printf("running alignment kernel\n");
//testforceDirectedKernel<<<(int)(g->nodes/256)+1,256>>>(g->nodes,g->coords_d,g->coinfo_d,g->edgeMatrix_d);
#ifdef STACKVIEW
printf("Running stack kernel\n");
//forceAlignStackKernel2d<<<(int)( algn->rows/256)+1,256>>>(algn->rows,algn->cols,algn->g1->coords_d,algn->g2->coords_d,algn->g1->coinfo_d,algn->g2->coinfo_d,algn->edgeAlignMatrix_d);
forceAlignStackKernel2d<<<(int)((algn->rows+0)/256)+1,256>>>(algn->rows,algn->cols,algn->g1->coords_d,algn->g2->coords_d,algn->g1->coinfo_d,algn->g2->coinfo_d,algn->edgeAlignMatrix_d);
//forceAlignStackKernel2d<<<(int)((algn->cols+0)/256)+1,256>>>(algn->cols,algn->rows,algn->g2->coords_d,algn->g1->coords_d,algn->g2->coinfo_d,algn->g2->coinfo_d,algn->edgeAlignMatrix_d);
#else
printf("Running Normal Kernel\n");
forceAlignKernel2d<<<(int)( algn->rows/256)+1,256>>>(algn->rows,algn->cols,algn->g1->coords_d,algn->g2->coords_d,algn->edgeAlignMatrix_d);
#endif
//cudaDeviceSynchronize();
//cudaMemcpy(algn->g1->coords,algn->g1->coords_d,algn->g1->nodes * sizeof(float)*3, cudaMemcpyDeviceToHost);
//cudaMemcpy(algn->g2->coords,algn->g2->coords_d,algn->g2->nodes * sizeof(float)*3, cudaMemcpyDeviceToHost);
return cudaGetLastError();
}
