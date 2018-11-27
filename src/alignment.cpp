#include "alignment.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>



Alignment::Alignment(const QString& filename, Graph *graph1, Graph *graph2):
	_graph1(graph1),
	_graph2(graph2)
{
	load_edges(filename);

	// initialize edge matrix
	_edge_matrix = Matrix(graph1->nodes().size(), graph2->nodes().size());
	_edge_matrix.init_zeros();

	for ( const edge_idx_t& edge : _edges )
	{
		int i = edge.node1;
		int j = edge.node2;

		_edge_matrix.elem(i, j) = 1;
	}

	// initialize vertices
	_vertices.resize(_edges.size());

	update();
}



/**
 * Load the algnment edge list from a file.
 *
 * @param filename
 */
void Alignment::load_edges(const QString& filename)
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

		int i = _graph1->find_node(node1);
		int j = _graph2->find_node(node2);

		if ( i == -1 )
		{
			qWarning() << "warning: could not find node " << node1;
		}

		if ( j == -1 )
		{
			qWarning() << "warning: could not find node " << node2;
		}

		if ( i != -1 && j != -1 )
		{
			_edges.push_back({ i, j });
		}
	}

	qInfo() << "- Loaded" << _edges.size() << "edges.";
}



/**
 * Save the algnment edge list to a file.
 *
 * @param filename
 */
void Alignment::save_edges(const QString& filename)
{
	qInfo() << "- Saving edges...";

	QFile file(filename);

	if ( !file.open(QIODevice::WriteOnly) )
	{
		qWarning("warning: unable to open edge file");
		return;
	}

	QTextStream out(&file);

	for ( const edge_idx_t& edge : _edges )
	{
		out << _graph1->nodes()[edge.node1].name
			<< "\t"
			<< _graph2->nodes()[edge.node2].name
			<< "\n";
	}

	qInfo() << "- Saved" << _edges.size() << "edges.";
}



/**
 * Extract the conserved subgraphs from an alignment of two graphs.
 *
 * Each conserved subgraph consists of only the nodes that have
 * an alignment edge. The graph edges between these nodes are also
 * extracted.
 */
void Alignment::extract_subgraphs()
{
	qInfo() << "Extracting subgraphs...";

	Graph g1;
	Graph g2;
	Alignment a1;
	Alignment a2;

	// extract nodes
	g1.nodes().reserve(_edges.size());
	g2.nodes().reserve(_edges.size());

	for ( const edge_idx_t& edge : _edges )
	{
		g1.nodes().push_back(_graph1->nodes()[edge.node1]);
		g2.nodes().push_back(_graph2->nodes()[edge.node2]);
	}

	g1.init_node_map();
	g2.init_node_map();

	// extract edges
	for ( const edge_idx_t& edge : _graph1->edges() )
	{
		int i = g1.find_node(_graph1->nodes()[edge.node1].name);
		int j = g1.find_node(_graph1->nodes()[edge.node2].name);

		if ( i != -1 && j != -1 )
		{
			g1.edges().push_back(edge_idx_t { i, j });
		}
	}

	for ( const edge_idx_t& edge : _graph2->edges() )
	{
		int i = g2.find_node(_graph2->nodes()[edge.node1].name);
		int j = g2.find_node(_graph2->nodes()[edge.node2].name);

		if ( i != -1 && j != -1 )
		{
			g2.edges().push_back(edge_idx_t { i, j });
		}
	}

	// extract alignment edges
	a1._graph1 = a1._graph2 = &g1;
	a2._graph1 = a2._graph2 = &g2;

	a1._edges.reserve(_edges.size());
	a2._edges.reserve(_edges.size());

	for ( const edge_idx_t& edge : _edges )
	{
		int i = g1.find_node(_graph1->nodes()[edge.node1].name);
		int j = g2.find_node(_graph2->nodes()[edge.node2].name);

		a1._edges.push_back(edge_idx_t { i, i });
		a2._edges.push_back(edge_idx_t { j, j });
	}

	// save graphs and alignments
	g1.save_nodes(_graph1->name() + "-cons_nodes.txt");
	g1.save_edges(_graph1->name() + "-cons_edges.txt");

	g2.save_nodes(_graph2->name() + "-cons_nodes.txt");
	g2.save_edges(_graph2->name() + "-cons_edges.txt");

	a1.save_edges(_graph1->name() + "-" + _graph1->name() + ".gna");
	a2.save_edges(_graph2->name() + "-" + _graph2->name() + ".gna");
}



/**
 * Update the vertices of each alignment edge to
 * the positions of the corresponding graph nodes.
 */
void Alignment::update()
{
	for ( int k = 0; k < _edges.size(); k++ )
	{
		int i = _edges[k].node1;
		int j = _edges[k].node2;

		_vertices[k].v1 = _graph1->positions()[i];
		_vertices[k].v2 = _graph2->positions()[j];
	}
}



void Alignment::print() const
{
	qInfo() << _graph1->name() << _graph2->name();

	for ( const edge_idx_t& edge : _edges )
	{
		qDebug() << edge.node1 << edge.node2;
	}
}
