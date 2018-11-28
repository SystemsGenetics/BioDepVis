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
	float x, float y, float z, float w, float h):
	_id(id),
	_name(name),
	_center({ x, y, z }),
	_width(w),
	_height(h)
{
	load_nodes(nodefile);
	load_edges(edgefile);

	if ( ontfile != "" )
	{
		load_ontology(ontfile);
	}

	// initialize positions
	_positions.reserve(_nodes.size());

	for ( int i = 0; i < _nodes.size(); i++ )
	{
		_positions.push_back({
			x - w / 2 + w * qrand() / RAND_MAX,
			y - h / 2 + h * qrand() / RAND_MAX,
			z
		});
	}

	// initialize delta positions
	_velocities.reserve(_nodes.size());

	for ( int i = 0; i < _nodes.size(); i++ )
	{
		_velocities.push_back({ 0, 0, 0 });
	}

	// determine number of modules
	int num_modules = 0;

	for ( const Node& node : _nodes )
	{
		if ( num_modules < node.module_id )
		{
			num_modules = node.module_id;
		}
	}

	// initialize colors
	_colors.reserve(_nodes.size());

	for ( const Node& node : _nodes )
	{
		QColor c = QColor::fromHsvF(
			(float) node.module_id / num_modules,
			0.8f,
			0.8f,
			1.0f
		);

		_colors.push_back({
			(float) c.redF(),
			(float) c.greenF(),
			(float) c.blueF(),
			(float) c.alphaF()
		});
	}

	// initialize edge matrix
	_edge_matrix = Matrix(_nodes.size(), _nodes.size());
	_edge_matrix.init_zeros();

	for ( const Edge& edge : _edges )
	{
		int i = edge.node1;
		int j = edge.node2;

		_edge_matrix.elem(i, j) = 1;
		_edge_matrix.elem(j, i) = 1;
	}

	// initialize GPU data
	CUDA_SAFE_CALL(cudaMalloc(&_positions_gpu, _positions.size() * sizeof(Vector3)));
	CUDA_SAFE_CALL(cudaMalloc(&_velocities_gpu, _velocities.size() * sizeof(Vector3)));
	gpu_write_positions();
	gpu_write_velocities();

	int n = _nodes.size();
	CUDA_SAFE_CALL(cudaMalloc(&_edge_matrix_gpu, (int64_t)n * n * sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpyAsync(
		_edge_matrix_gpu,
		_edge_matrix.data(),
		(int64_t)n * n * sizeof(bool),
		cudaMemcpyHostToDevice));
}



Graph::~Graph()
{
	CUDA_SAFE_CALL(cudaFree(_positions_gpu));
	CUDA_SAFE_CALL(cudaFree(_velocities_gpu));
	CUDA_SAFE_CALL(cudaFree(_edge_matrix_gpu));
}



void Graph::gpu_read_positions()
{
	CUDA_SAFE_CALL(cudaMemcpyAsync(
		_positions.data(),
		_positions_gpu,
		_positions.size() * sizeof(Vector3),
		cudaMemcpyDeviceToHost));
}



void Graph::gpu_read_velocities()
{
	CUDA_SAFE_CALL(cudaMemcpyAsync(
		_velocities.data(),
		_velocities_gpu,
		_velocities.size() * sizeof(Vector3),
		cudaMemcpyDeviceToHost));
}



void Graph::gpu_write_positions()
{
	CUDA_SAFE_CALL(cudaMemcpyAsync(
		_positions_gpu,
		_positions.data(),
		_positions.size() * sizeof(Vector3),
		cudaMemcpyHostToDevice));
}



void Graph::gpu_write_velocities()
{
	CUDA_SAFE_CALL(cudaMemcpyAsync(
		_velocities_gpu,
		_velocities.data(),
		_velocities.size() * sizeof(Vector3),
		cudaMemcpyHostToDevice));
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

	for ( int i = 0; i < _nodes.size(); i++ )
	{
		const Node& node = _nodes[i];

		if ( !_node_map.contains(node.name) )
		{
			_node_map.insert(node.name, i);
		}
		else
		{
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

	if ( !file.open(QIODevice::ReadOnly) )
	{
		qWarning("warning: unable to open node file");
		return;
	}

	QTextStream in(&file);

	while ( !in.atEnd() )
	{
		QStringList list = in.readLine().split("\t");
		QString name = list[0];
		int module_id = list[1].toInt();

		Node node;
		node.name = name;
		node.module_id = module_id;

		if ( !_node_map.contains(name) )
		{
			_nodes.push_back(node);
			_node_map.insert(node.name, _nodes.size() - 1);
		}
		else
		{
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

	if ( !file.open(QIODevice::ReadOnly) )
	{
		qWarning("warning: unable to open edge file");
		return;
	}

	QTextStream in(&file);

	while ( !in.atEnd() )
	{
		QStringList list = in.readLine().split("\t");
		QString node1 = list[0];
		QString node2 = list[1];

		int i = find_node(node1);
		int j = find_node(node2);

		if ( i != -1 && j != -1 )
		{
			_edges.push_back({ i, j });
		}
		else
		{
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

	if ( !file.open(QIODevice::ReadOnly) )
	{
		qWarning("warning: unable to open ontology file");
		return;
	}

	QTextStream in(&file);

	while ( !in.atEnd() )
	{
		QStringList fields = in.readLine().split("\t");
		QString name = fields[1];
		QStringList go_terms = fields[9].split(",");

		int nodeIndex = find_node(name);

		if ( nodeIndex != -1 )
		{
			_nodes[nodeIndex].go_terms = go_terms;
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

	if ( !file.open(QIODevice::WriteOnly) )
	{
		qWarning("warning: unable to open node file");
		return;
	}

	QTextStream out(&file);

	for ( const Node& node : _nodes )
	{
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

	if ( !file.open(QIODevice::WriteOnly) )
	{
		qWarning("warning: unable to open edge file");
		return;
	}

	QTextStream out(&file);

	for ( const Edge& edge : _edges )
	{
		out << _nodes[edge.node1].name
			<< "\t"
			<< _nodes[edge.node2].name
			<< "\n";
	}

	qInfo() << "- Saved" << _edges.size() << "edges.";
}



void Graph::print() const
{
	qInfo() << _id << _name;

	for ( int i = 0; i < _nodes.size(); i++ )
	{
		qDebug()
			<< _nodes[i].name
			<< _nodes[i].module_id
			<< _nodes[i].go_terms.join(' ');
	}
}
