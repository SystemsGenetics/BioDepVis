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

    load_alignment(filename);
}

Alignment::Alignment()
{
    this->_graph1 = nullptr;
    this->_graph2 = nullptr;
    this->_edge_matrix = nullptr;
}

Alignment::~Alignment()
{
    // delete[] this->_edge_matrix;
}

bool Alignment::load_alignment(const QString& filename)
{
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
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
            qDebug() << "warning: could not find nodes " << node1 << node2;
        }
    }

    // initialize edge matrix
    for ( auto& edge : this->_edges ) {
        int i = edge.first;
        int j = edge.second;

        this->_edge_matrix[i * this->_cols + j] = 1;
    }

    return true;
}

void Alignment::print() const
{
    qDebug() << this->_graph1->name() << this->_graph2->name();
    qDebug() << "";

    for ( auto& edge : this->_edges ) {
        qDebug() << edge.first << edge.second;
    }
}
