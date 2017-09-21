#include "graph.h"
#include <QDebug>
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

    //initialize the coord info struct for clusterization
    coinfo.reserve(nodes);
    for(int i = 0; i < nodes;i++)
    {
        coord_t c = {
            0,0,0,1,0
        };
        coinfo.push_back(c);
    }

    load_clusterfile(clusterfile);
	load_ontologyfile(ontfile);
}

/**
 * Search a list of nodes by name.
 *
 * @param nodes
 * @param name
 */
int find_index(const QVector<node_t>& nodes, const QString& name)
{
	for ( int i = 0; i < nodes.size(); i++ ) {
		if ( nodes[i].name == name ) {
			return i;
		}
	}

	return -1;
}

bool Graph::load_datafile(const QString& filename){

    QFile file(filename);
    QTextStream in(&file);
    QString line;
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

         if ( find_index(nodeList, node1) == -1 ) {
			 node_t n;
			 n.name = node1;

             nodeList.push_back(n);
         }

         if ( find_index(nodeList, node2) == -1 ) {
			 node_t n;
			 n.name = node2;

            nodeList.push_back(n);
         }

    }
    nodes = nodeList.size();
    edgeMatrix = new float[nodes * nodes];
    edges=0;

    for(std::pair<QString,QString> p : nodePairs){

        node1 = p.first;
		node2 = p.second;
        int i = find_index(nodeList, node1);
        int j = find_index(nodeList, node2);

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

        int nodeIndex = find_index(nodeList, node);
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

        int nodeIndex = find_index(nodeList, name);

		if ( nodeIndex != -1 ) {
			nodeList[nodeIndex].go_terms = goTermList;
		}
     }

}

void Graph::print() const
{
	qDebug() << this->_id << this->_name;

	for ( int i = 0; i < this->nodeList.size(); i++ ) {
		qDebug()
			<< this->nodeList[i].name
			<< this->coinfo[i].clusterId
			<< this->nodeList[i].go_terms.join(' ');
	}
}
