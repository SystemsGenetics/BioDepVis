#include "graph.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>

Graph::Graph(
    int id, const QString& name,
    const QString& nodefile,
    const QString& edgefile,
    const QString& ontfile,
    float x, float y, float z, int w, int h)
{
    this->_id = id;
    this->_name = name;
    this->_center = { x, y, z };
    this->_width = w;
    this->_height = h;

    load_nodes(nodefile);
    load_edges(edgefile);
    load_ontology(ontfile);

    // initialize coords
    this->_coords.reserve(this->_nodes.size());

    for ( int i = 0; i < this->_nodes.size(); i++ ) {
        this->_coords.push_back({
            x - w/2 + qrand() % w,
            y - w/2 + qrand() % h,
            z
        });
    }

    // initialize coinfo
    this->_coinfo.reserve(this->_nodes.size());

    for ( int i = 0; i < this->_nodes.size(); i++ ) {
        this->_coinfo.push_back({ 0, 0, 0, 1 });
    }
}

Graph::Graph()
{
    this->_edge_matrix = nullptr;
}

Graph::~Graph()
{
    delete[] this->_edge_matrix;
}

/**
 * Find a node by name.
 *
 * @param name
 */
int Graph::find_node(const QString& name)
{
    for ( int i = 0; i < this->_nodes.size(); i++ ) {
        if ( this->_nodes[i].name == name ) {
            return i;
        }
    }

    return -1;
}

void Graph::load_nodes(const QString& filename)
{
    qInfo() << "- loading nodes...";

    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        qWarning("warning: unable to open node file");
        return;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString name = list[0];
        int module_id = list[1].toInt();

        graph_node_t node;
        node.name = name;
        node.module_id = module_id;

        this->_nodes.push_back(node);
    }
}

void Graph::load_edges(const QString& filename)
{
    qInfo() << "- loading edges...";

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

        int i = this->find_node(node1);
        int j = this->find_node(node2);

        if ( i != -1 && j != -1 ) {
            this->_edges.push_back({ i, j });
        }
        else {
            qWarning() << "warning: could not find nodes " << node1 << node2;
        }
    }

    // initialize edge matrix
    int rows = this->_nodes.size();

    this->_edge_matrix = new float[rows * rows];

    memset(this->_edge_matrix, 0, rows * rows * sizeof(float));

    for ( const graph_edge_t& edge : this->_edges ) {
        int i = edge.node1;
        int j = edge.node2;

        this->_edge_matrix[i * rows + j] = 1;
        this->_edge_matrix[j * rows + i] = 1;
    }
}

void Graph::load_ontology(const QString& filename)
{
    qInfo() << "- loading ontology...";

    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        qWarning("warning: unable to open ontology file");
        return;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList fields = in.readLine().split("\t");
        QString name = fields[1];
        QStringList go_terms = fields[9].split(",");

        int nodeIndex = this->find_node(name);

        if ( nodeIndex != -1 ) {
            this->_nodes[nodeIndex].go_terms = go_terms;
        }
    }
}

void Graph::print() const
{
    qInfo() << this->_id << this->_name;

    // for ( int i = 0; i < this->_nodes.size(); i++ ) {
    //     qDebug()
    //         << this->_nodes[i].name
    //         << this->_nodes[i].module_id
    //         << this->_nodes[i].go_terms.join(' ');
    // }
}
