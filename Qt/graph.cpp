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

    //initialize the coord info struct for clusterization
    coinfo.reserve(nodes);
    for(int i = 0; i < nodes;i++)
    {
        coord_t c = {
            0,0,0,1,0
        };
        coinfo.push_back(c);
    }

    load_datafile(datafile);
    load_clusterfile(clusterfile);


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
    goTerm.reserve(nodes);
    edgeMatrix = new float[nodes * nodes];
    edges=0;

    for(std::pair<QString,QString> p : nodePairs){

        node1 = p.first;
        int i = nodeListMap.indexOf(node1);
        int j = nodeListMap.indexOf(node2);

        edgeMatrix[i*nodes + j]=1;
        edgeMatrix[j*nodes + i]=1;

        edges++;

    }

}
bool Graph::load_clusterfile(const QString& filename){

    QFile file(filename);
    QTextStream in(&file);
    QString line;
    QString node;//gene?
    int clusterId;

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    while(!in.atEnd()){
        line = in.readLine();
        QStringList list = line.split("\t");
        node = list[0];
        clusterId =list[1].toInt();

        int nodeIndex = nodeListMap.indexOf(node);
        coinfo[nodeIndex].clusterId = clusterId;
    }

}

bool Graph::load_ontologyfile(const QString& filename){
    QFile file(filename);
    QTextStream in(&file);
    QString line;
    QString name;
    QStringList goTermList;


    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

     while(!in.atEnd()){
         line = in.readLine();
         QStringList lineContents = line.split("\t");
         name = lineContents[1];
         goTermList = lineContents[9].split(",");

         int nodeIndex = nodeListMap.indexOf(name);
         //goTerm.assign(nodeIndex, goTermList);
     }

}



