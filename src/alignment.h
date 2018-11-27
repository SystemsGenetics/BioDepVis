#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include "graph.h"



typedef struct
{
	vec3_t v1;
	vec3_t v2;
} edge_vtx_t;



class Alignment
{
private:
	Graph *_graph1 {nullptr};
	Graph *_graph2 {nullptr};
	QVector<edge_idx_t> _edges;
	QVector<edge_vtx_t> _vertices;
	Matrix _edge_matrix;

public:
	Alignment(const QString& filename, Graph *graph1, Graph *graph2);
	Alignment() = default;

	const QVector<edge_idx_t>& edges() const { return _edges; }
	const QVector<edge_vtx_t>& vertices() const { return _vertices; }

	void load_edges(const QString& filename);
	void save_edges(const QString& filename);
	void extract_subgraphs();

	void update();

	void print() const;
};



#endif // ALIGNMENT_H
