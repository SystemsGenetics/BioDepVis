#include "graph.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>

const float CLUSTER_SIZE = 360.0f;
const float NODE_ALPHA = 0.6f;

color_t hsv_to_rgb(float h, float s, float v)
{
    int i;
    float f, p, q, t;

    if ( s == 0 ) {
        return (color_t) { v, v, v, 0 };
    }

    h /= 60;
    i = floor(h);
    f = h - i;
    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));

    switch ( i ) {
    case 0:
        return (color_t) { v, t, p, 0 };
    case 1:
        return (color_t) { q, v, p, 0 };
    case 2:
        return (color_t) { p, v, t, 0 };
    case 3:
        return (color_t) { p, q, v, 0 };
    case 4:
        return (color_t) { t, p, v, 0 };
    case 5:
    default:
        return (color_t) { v, p, q, 0 };
    }
}

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

    // initialize colors
    this->_colors.reserve(this->_nodes.size());

    for ( const graph_node_t& node : this->_nodes ) {
        float h = node.module_id / CLUSTER_SIZE * 360.0f;
        float s = 0.8f;
        float v = 1.0f - node.module_id / CLUSTER_SIZE;

        color_t color = hsv_to_rgb(h, s, v);
        color.a = NODE_ALPHA;

        this->_colors.push_back(color);
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
