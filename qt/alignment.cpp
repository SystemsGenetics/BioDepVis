#include "alignment.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>

Alignment::Alignment(const QString& filename, Graph *graph1, Graph *graph2)
{
    this->_graph1 = graph1;
    this->_graph2 = graph2;

    load_edges(filename);

    // initialize edge matrix
    this->_rows = graph1->nodes().size();
    this->_cols = graph2->nodes().size();
    this->_edge_matrix = new float[this->_rows * this->_cols];

    memset(this->_edge_matrix, 0, this->_rows * this->_cols * sizeof(float));

    for ( const graph_edge_t& edge : this->_edges ) {
        int i = edge.node1;
        int j = edge.node2;

        this->_edge_matrix[i * this->_cols + j] = 1;
    }

    // initialize vertices
    this->_vertices.reserve(this->_edges.size());
}

Alignment::Alignment()
{
    this->_graph1 = nullptr;
    this->_graph2 = nullptr;
    this->_edge_matrix = nullptr;
}

Alignment::~Alignment()
{
    delete[] this->_edge_matrix;
}

/**
 * Load the algnment edge list from a file.
 *
 * @param filename
 */
void Alignment::load_edges(const QString& filename)
{
    qInfo() << "- loading edges...";

    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        qWarning("warning: unable to open edge file");
        return;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString node1 = list[0];
        QString node2 = list[1];

        int i = this->_graph1->find_node(node1);
        int j = this->_graph2->find_node(node2);

        if ( i == -1 ) {
            qWarning() << "warning: could not find node " << node1;
        }
        if ( j == -1 ) {
            qWarning() << "warning: could not find node " << node2;
        }
        if ( i != -1 && j != -1 ) {
            this->_edges.push_back({ i, j });
        }
    }
}

/**
 * Update the vertices of each alignment edge to
 * the positions of the corresponding graph nodes.
 */
void Alignment::update()
{
    for ( int k = 0; k < this->_edges.size(); k++ ) {
        int i = this->_edges[k].node1;
        int j = this->_edges[k].node2;

        this->_vertices[k].v1 = this->_graph1->coords()[i];
        this->_vertices[k].v2 = this->_graph2->coords()[j];
    }
}

void Alignment::print() const
{
    qInfo() << this->_graph1->name() << this->_graph2->name();

    // for ( const graph_edge_t& edge : this->_edges ) {
    //     qDebug() << edge.node1 << edge.node2;
    // }
}
