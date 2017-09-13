#include "graph.h"
#include <QFile>
#include<QTextStream>

Graph::Graph(
	int id, const QString& name,
	const QString& datafile,
	const QString& clusterfile,
	const QString& ontfile,
	int x, int y, int z, int w, int h)
{
	this->_id = id;
	this->_name = name;
    load_datafile(datafile);
}

bool Graph::load_datafile(const QString& filename){

    QFile file(filename);
    QTextStream in(&file);
    QString line;
    in.readLine();//for fist line
    QString node1;
    QString node2;

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    std::vector<std::pair<QString,QString>> nodePairs;
     while(!in.atEnd()){
         line = in.readLine();
         QStringList list = line.split("\t");
         node1 = list[0];
         node2 = list[1];
         nodePairs.push_back({node1,node2});//might work
     }

     for( std::pair<QString,QString> p : nodePairs){
         node1 = p.first;
         node2 = p.second;

         bool found = nodeListMap.contains(node1);
         if (!found){
             nodeListMap.push_back(node1);
         }

        bool found2 = nodeListMap.contains(node2);
         if (!found2){
            nodeListMap.push_back(node2);
         }

    }
    nodes = nodeListMap.size();
    goTerm = new std::vector<std::string>[nodes];
    edgeMatrix = new float[nodes * nodes];
    edges=0;

    for(std::pair<QString,QString> p : nodePairs){

        node1 = p.first;
        int i = 0;
        int j = 0;

        for(int i = 0;i<nodes;i++){
            if(nodeListMap[i] == node1 ){
                break;
            }
        }

        for(int j = 0;j<nodes;j++){
            if(nodeListMap[j] == node1 ){
                break;
            }
        }

        edgeMatrix[i*nodes + j]=1;
        edgeMatrix[j*nodes + i]=1;

        edges++;

    }


}



