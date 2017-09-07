#include "graph.h"

Graph::Graph(
	int id, const QString& name,
	const QString& datafile,
	const QString& clusterfile,
	const QString& ontfile,
	int x, int y, int z, int w, int h)
{
	this->_id = id;
	this->_name = name;
}
