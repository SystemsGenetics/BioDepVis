#ifndef DATABASE_H
#define DATABASE_H

#include <QHash>
#include <QMap>
#include "alignment.h"



struct NodeRef
{
	int graph_id;
	int node_id;
};



struct OntologyTerm
{
	QString id;
	QString name;
	QString def;
	QVector<NodeRef> connected_nodes;
};



class Database
{
private:
	QMap<int, Graph *> _graphs;
	QVector<Alignment *> _alignments;
	QHash<QString, OntologyTerm> _ontology;

public:
	Database() = default;

	QMap<int, Graph *>& graphs() { return _graphs; }
	QVector<Alignment *>& alignments() { return _alignments; }
	const QHash<QString, OntologyTerm>& ontology() const { return _ontology; }

	Node& node(const NodeRef& ref) { return _graphs[ref.graph_id]->nodes()[ref.node_id]; }

	void load_config(const QString& filename);
	void load_ontology(const QString& filename);

	void print() const;
};



#endif
