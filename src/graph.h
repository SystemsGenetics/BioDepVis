#ifndef GRAPH_H
#define GRAPH_H

#include <QHash>
#include <QString>
#include <QVector>
#include "matrix.h"
#include "vector.h"



struct Color
{
	float r;
	float g;
	float b;
	float a;
};



struct Node
{
	QString name;
	int module_id;
	QStringList go_terms;
};



struct Edge
{
	int node1;
	int node2;
};



class Graph
{
private:
	int _id;
	QString _name;
	Vector3 _center;

	QVector<Node> _nodes;
	QHash<QString, int> _node_map;
	QVector<Edge> _edges;
	QVector<Color> _colors;

	QVector<Vector3> _positions;
	QVector<Vector3> _velocities;
	Matrix _edge_matrix;

	Vector3 *_positions_gpu {nullptr};
	Vector3 *_velocities_gpu {nullptr};
	bool *_edge_matrix_gpu {nullptr};

public:
	Graph(
		int id, const QString& name,
		const QString& nodefile,
		const QString& edgefile,
		const QString& ontfile,
		float x, float y, float z,
		float radius
	);
	Graph() = default;
	~Graph();

	int id() const { return _id; }
	const QString& name() const { return _name; }
	QVector<Node>& nodes() { return _nodes; }
	QVector<Edge>& edges() { return _edges; }
	QVector<Color>& colors() { return _colors; }

	QVector<Vector3>& positions() { return _positions; }
	QVector<Vector3>& velocities() { return _velocities; }
	Matrix& edge_matrix() { return _edge_matrix; }

	Vector3 * positions_gpu() { return _positions_gpu; }
	Vector3 * velocities_gpu() { return _velocities_gpu; }
	bool * edge_matrix_gpu() const { return _edge_matrix_gpu; }

	void gpu_read_positions();
	void gpu_read_velocities();
	void gpu_write_positions();
	void gpu_write_velocities();

	int find_node(const QString& name);

	void init_node_map();
	void load_nodes(const QString& filename);
	void load_edges(const QString& filename);
	void load_ontology(const QString& filename);

	void save_nodes(const QString& filename);
	void save_edges(const QString& filename);

	void print() const;
};



#endif // GRAPH
