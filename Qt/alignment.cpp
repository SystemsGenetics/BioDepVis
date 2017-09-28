#include "alignment.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>

Alignment::Alignment(const QString& filename, Graph *graph1, Graph *graph2)
{
    this->_graph1 = graph1;
    this->_graph2 = graph2;

    this->_rows = graph1->nodes().size();
    this->_cols = graph2->nodes().size();
    this->_edge_matrix = new float[this->_rows * this->_cols];

    memset(this->_edge_matrix, 0, this->_rows * this->_cols * sizeof(float));

    load_edges(filename);
}

Alignment::Alignment()
{
    this->_graph1 = nullptr;
    this->_graph2 = nullptr;
    this->_edge_matrix = nullptr;
}

Alignment::~Alignment()
{
    // TODO: causes segfault, need copy constructor?
    // delete[] this->_edge_matrix;
}

void Alignment::load_edges(const QString& filename)
{
    qInfo() << "- loading edge list...";

    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        qWarning("warning: unable to open edge file");
        return;
    }

    // load edges from file
    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString node1 = list[0];
        QString node2 = list[1];

        int i = find_node(this->_graph1->nodes(), node1);
        int j = find_node(this->_graph2->nodes(), node2);

        if ( i != -1 && j != -1 ) {
            this->_edges.push_back({ i, j });
        }
        else {
            qWarning() << "warning: could not find nodes " << node1 << node2;
        }
    }

    // initialize edge matrix
    for ( const graph_edge_t& edge : this->_edges ) {
        int i = edge.node1;
        int j = edge.node2;

        this->_edge_matrix[i * this->_cols + j] = 1;
    }

    // initialize vertices
    this->_vertices.reserve(this->_edges.size());
}

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

    // for ( auto& edge : this->_edges ) {
    //     qDebug() << edge.first << edge.second;
    // }
}
