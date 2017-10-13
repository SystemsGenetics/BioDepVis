#include <QColor>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include "fdl.h"
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

    if ( ontfile != "" ) {
        load_ontology(ontfile);
    }

    // initialize positions
    this->_positions.reserve(this->_nodes.size());

    for ( int i = 0; i < this->_nodes.size(); i++ ) {
        this->_positions.push_back({
            x - w / 2 + w * qrand() / RAND_MAX,
            y - h / 2 + h * qrand() / RAND_MAX,
            z
        });
    }

    // initialize delta positions
    this->_positions_d.reserve(this->_nodes.size());

    for ( int i = 0; i < this->_nodes.size(); i++ ) {
        this->_positions_d.push_back({ 0, 0, 0 });
    }

    // determine number of modules
    int num_modules = 0;

    for ( const node_t& node : this->_nodes ) {
        if ( num_modules < node.module_id ) {
            num_modules = node.module_id;
        }
    }

    // initialize colors
    this->_colors.reserve(this->_nodes.size());

    for ( const node_t& node : this->_nodes ) {
        QColor c = QColor::fromHsvF(
            (float) node.module_id / num_modules,
            0.8f,
            0.8f,
            1.0f
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

    for ( const edge_idx_t& edge : this->_edges ) {
        int i = edge.node1;
        int j = edge.node2;

        this->_edge_matrix.elem(i, j) = 1;
        this->_edge_matrix.elem(j, i) = 1;
    }

    // initialize GPU data
    int n = _nodes.size();
    _positions_gpu = (vec3_t *)gpu_malloc(n * sizeof(vec3_t));
    _positions_d_gpu = (vec3_t *)gpu_malloc(n * sizeof(vec3_t));
    _edge_matrix_gpu = (bool *)gpu_malloc(n * n * sizeof(bool));

    write_gpu();
}

Graph::Graph()
{
    _positions_gpu = nullptr;
    _positions_d_gpu = nullptr;
    _edge_matrix_gpu = nullptr;
}

Graph::~Graph()
{
    gpu_free(_positions_gpu);
    gpu_free(_positions_d_gpu);
    gpu_free(_edge_matrix_gpu);
}

void Graph::read_gpu()
{
    int n = _nodes.size();
    gpu_read(_positions.data(), _positions_gpu, n * sizeof(vec3_t));
    gpu_read(_positions_d.data(), _positions_d_gpu, n * sizeof(vec3_t));
    gpu_read(_edge_matrix.data(), _edge_matrix_gpu, n * n * sizeof(bool));
}

void Graph::write_gpu()
{
    int n = _nodes.size();
    gpu_write(_positions_gpu, _positions.data(), n * sizeof(vec3_t));
    gpu_write(_positions_d_gpu, _positions_d.data(), n * sizeof(vec3_t));
    gpu_write(_edge_matrix_gpu, _edge_matrix.data(), n * n * sizeof(bool));
}

/**
 * Find a node by name.
 *
 * @param name
 */
int Graph::find_node(const QString& name)
{
    return _node_map.contains(name)
	? _node_map[name]
	: -1;
}

void Graph::init_node_map()
{
    _node_map.clear();

    for ( int i = 0; i < _nodes.size(); i++ ) {
        const node_t& node = _nodes[i];

        if ( !_node_map.contains(node.name) ) {
            _node_map.insert(node.name, i);
        }
        else {
            qWarning() << "warning: duplicate node" << node.name;
        }
    }
}

/**
 * Load the node list from a file.
 *
 * @param filename
 */
void Graph::load_nodes(const QString& filename)
{
    qInfo() << "- Loading nodes...";

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

        node_t node;
        node.name = name;
        node.module_id = module_id;

        if ( !_node_map.contains(name) ) {
            _nodes.push_back(node);
            _node_map.insert(node.name, _nodes.size() - 1);
        }
        else {
            qWarning() << "warning: duplicate node" << name;
        }
    }

    qInfo() << "- Loaded" << _nodes.size() << "nodes.";
}

/**
 * Load the edge list from a file.
 *
 * @param filename
 */
void Graph::load_edges(const QString& filename)
{
    qInfo() << "- Loading edges...";

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

    qInfo() << "- Loaded" << _edges.size() << "edges.";
}

/**
 * Load the ontology terms list from a file.
 *
 * @param filename
 */
void Graph::load_ontology(const QString& filename)
{
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

/**
 * Save the node list to a file.
 *
 * @param filename
 */
void Graph::save_nodes(const QString& filename)
{
    qInfo() << "- Saving nodes...";

    QFile file(filename);

    if ( !file.open(QIODevice::WriteOnly) ) {
        qWarning("warning: unable to open node file");
        return;
    }

    QTextStream out(&file);

    for ( const node_t& node : _nodes ) {
        out << node.name
            << "\t" << node.module_id
            << "\t" << node.go_terms.join(',')
            << "\n";
    }

    qInfo() << "- Saved" << _nodes.size() << "nodes.";
}

/**
 * Save the edge list to a file.
 *
 * @param filename
 */
void Graph::save_edges(const QString& filename)
{
    qInfo() << "- Saving edges...";

    QFile file(filename);

    if ( !file.open(QIODevice::WriteOnly) ) {
        qWarning("warning: unable to open edge file");
        return;
    }

    QTextStream out(&file);

    for ( const edge_idx_t& edge : _edges ) {
        out << _nodes[edge.node1].name
            << "\t"
            << _nodes[edge.node2].name
            << "\n";
    }

    qInfo() << "- Saved" << _edges.size() << "edges.";
}

void Graph::print() const
{
    qInfo() << this->_id << this->_name;

    for ( int i = 0; i < this->_nodes.size(); i++ ) {
        qDebug()
            << this->_nodes[i].name
            << this->_nodes[i].module_id
            << this->_nodes[i].go_terms.join(' ');
    }
}
