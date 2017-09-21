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

bool Graph::load_datafile(const QString& filename)
{
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    // read node pairs from file
    QTextStream in(&file);
    std::vector<std::pair<QString,QString>> nodePairs;

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString node1 = list[0];
        QString node2 = list[1];

        nodePairs.push_back({ node1, node2 });
    }

    // construct node list from pairs
    for ( auto& p : nodePairs ) {
        QString node1 = p.first;
        QString node2 = p.second;

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

    // construct edge matrix
    nodes = nodeList.size();
    edgeMatrix = new float[nodes * nodes];
    edges=0;

    for ( auto& p : nodePairs ) {
        QString node1 = p.first;
        QString node2 = p.second;
        int i = find_index(nodeList, node1);
        int j = find_index(nodeList, node2);

        edgeMatrix[i*nodes + j] = 1;
        edgeMatrix[j*nodes + i] = 1;

        edges++;
    }

    return true;
}

bool Graph::load_clusterfile(const QString& filename)
{
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString node = list[0];
        int clusterId = list[1].toInt();

        int nodeIndex = find_index(nodeList, node);
        coinfo[nodeIndex].clusterId = clusterId;
    }

    return true;
}

bool Graph::load_ontologyfile(const QString& filename)
{
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList fields = in.readLine().split("\t");
        QString name = fields[1];
        QStringList go_terms = fields[9].split(",");

        int nodeIndex = find_index(nodeList, name);

        if ( nodeIndex != -1 ) {
            nodeList[nodeIndex].go_terms = go_terms;
        }
    }

    return true;
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
