#include <math.h>
#include "graph.h"

#define MAX_DISPLACEMENT_SQUARED 10.0f
void forceDirectedLayout(float *nodePos,float *nodePosD,int nodes, float *matrixEdge)
{
 
    printf("Runnign Force Directed\n");
    float K_r = 25.0;
    float K_s = 15.0f;
    float L = 1.2f;
    float delta_t = 0.004f;
    int i,j;
    for(i = 0; i < nodes;i++)
    {
        for(j = 0;j < nodes;j++)
        {
           // float dz = nodePos[j*3+2] - nodePos[i*3+2];

            float dy = nodePos[j*3+1] - nodePos[i*3+1];
            float dx = nodePos[j*3+0] - nodePos[i*3+0];
            if(dx != 0.0f && dy!= 0.0f &&  i!=j)
            {
                //float distance  = sqrt(dx * dx + dy * dy + dz * dz);
                float distance  = sqrt(dx * dx + dy * dy )+0.001;

                float force2= 0.0f;
                float force = 0.0f;
                
                if(matrixEdge[i * nodes + j] < 1.0f)
                    force = K_r / (distance * distance);
                if(matrixEdge[i * nodes + j] >= 1.0f)
                    force2 =  K_s * (distance - L);
                
                
                nodePosD[i*INFOCOUNT+0] = nodePosD[i*INFOCOUNT+0]- ((force * dx)/distance) +((force2*dx)/distance) ;
                nodePosD[i*INFOCOUNT+1] = nodePosD[i*INFOCOUNT+1]- ((force * dy)/distance) +((force2*dy)/distance) ;

                nodePosD[j*INFOCOUNT+0] = nodePosD[j*INFOCOUNT+0]+ ((force * dx)/distance) -((force2*dx)/distance) ;
                nodePosD[j*INFOCOUNT+1] = nodePosD[j*INFOCOUNT+1]+ ((force * dy)/distance) -((force2*dy)/distance) ;

                
            }
            
        }
        
        
        float d_x = delta_t * nodePosD[i*INFOCOUNT+0];
        float d_y = delta_t * nodePosD[i*INFOCOUNT+1];
//        float d_z = delta_t * nodePosD[i*4+2];
//        float displacementSquared = d_x*d_x + d_y*d_y + d_z*d_z;
          float displacementSquared = d_x*d_x + d_y*d_y ;
        if ( displacementSquared > MAX_DISPLACEMENT_SQUARED ){
            float s = sqrt( MAX_DISPLACEMENT_SQUARED  / displacementSquared );
            d_x = d_x * s;
            d_y = d_y * s;
  //          d_z = d_z * s;
        }
  //      nodePos[i*3+2] += d_z;

        if(std::isnan(d_x)){
            d_x = 0.001;//printf("NAN");
        }
        if(std::isnan(d_y)){
            d_y = 0.001;//printf("NAN");
        }
        nodePos[i*3+1] += d_y;
        nodePos[i*3+0] += d_x;
        
        nodePosD[i*INFOCOUNT+0] *= .09f;
        nodePosD[i*INFOCOUNT+1] *= .09f;
    
      //  nodePosD[i*4+2] *= .06f;

    }
      //  i =1;
       // printf("%d = %f %f %f", 1,nodePos[i*3+0],nodePos[i*3+1],nodePos[i*3+2]);
    
}


