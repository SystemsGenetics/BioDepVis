#include <QColor>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include "graph.h"

Graph::Graph(
    int id, const QString& name,
    const QString& nodefile,
    const QString& edgefile,
    const QString& ontfile,
    float x, float y, float z, float w, float h)
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
            x - w / 2 + w * qrand() / RAND_MAX,
            y - h / 2 + h * qrand() / RAND_MAX,
            z
        });
    }

    // initialize delta coords
    this->_coords_d.reserve(this->_nodes.size());

    for ( int i = 0; i < this->_nodes.size(); i++ ) {
        this->_coords_d.push_back({ 0, 0, 0 });
    }

    // determine number of modules
    int num_modules = 0;

    for ( const graph_node_t& node : this->_nodes ) {
        if ( num_modules < node.module_id ) {
            num_modules = node.module_id;
        }
    }

    // initialize colors
    this->_colors.reserve(this->_nodes.size());

    for ( const graph_node_t& node : this->_nodes ) {
        QColor c = QColor::fromHsvF(
            (float) node.module_id / num_modules,
            0.8f,
            1.0f - (float) node.module_id / num_modules,
            0.6f
        );

        this->_colors.push_back({
            (float) c.redF(),
            (float) c.greenF(),
            (float) c.blueF(),
            (float) c.alphaF()
        });
    }

    // initialize edge matrix
    this->_edge_matrix = Matrix(this->_nodes.size(), this->_nodes.size());
    this->_edge_matrix.init_zeros();

    for ( const graph_edge_t& edge : this->_edges ) {
        int i = edge.node1;
        int j = edge.node2;

        this->_edge_matrix.elem(i, j) = 1;
        this->_edge_matrix.elem(j, i) = 1;
    }
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

/**
 * Load the node list from a file.
 *
 * @param filename
 */
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

/**
 * Load the edge list from a file.
 *
 * @param filename
 */
void Graph::load_edges(const QString& filename)
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

        int i = this->find_node(node1);
        int j = this->find_node(node2);

        if ( i != -1 && j != -1 ) {
            this->_edges.push_back({ i, j });
        }
        else {
            qWarning() << "warning: could not find nodes " << node1 << node2;
        }
    }
}

/**
 * Load the ontology terms list from a file.
 *
 * @param filename
 */
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
