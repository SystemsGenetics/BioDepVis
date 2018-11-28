#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include "graph.h"



struct Vertex
{
	Vector3 v1;
	Vector3 v2;
};



class Alignment
{
private:
	Graph *_graph1 {nullptr};
	Graph *_graph2 {nullptr};
	QVector<Edge> _edges;
	QVector<Vertex> _vertices;
	Matrix _edge_matrix;

public:
	Alignment(const QString& filename, Graph *graph1, Graph *graph2);
	Alignment() = default;

	const QVector<Edge>& edges() const { return _edges; }
	const QVector<Vertex>& vertices() const { return _vertices; }

	void load_edges(const QString& filename);
	void save_edges(const QString& filename);
	void extract_subgraphs();

	void update();

	void print() const;
};



#endif // ALIGNMENT_H
