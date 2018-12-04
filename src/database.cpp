#include <QDebug>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTextStream>
#include "database.h"



/**
 * Load graph and alignment data from a configuration file.
 *
 * @param filename
 */
void Database::load_config(const QString& filename)
{
	QFile file(filename);

	if ( !file.open(QIODevice::ReadOnly) )
	{
		qWarning("warning: unable to open config file");
		return;
	}

	QByteArray data = file.readAll();
	QJsonObject config = QJsonDocument::fromJson(data).object();

	qInfo() << "Loading graphs...";

	QJsonArray graphs = config["graphs"].toArray();

	for ( int i = 0; i < graphs.size(); i++ )
	{
		QJsonObject obj = graphs[i].toObject();

		qInfo() << obj["id"].toInt() << obj["name"].toString();

		Graph *g = new Graph(
			obj["id"].toInt(),
			obj["name"].toString(),
			obj["nodeFile"].toString(),
			obj["edgeFile"].toString(),
			obj["ontologyFile"].toString(),
			obj["x"].toDouble(),
			obj["y"].toDouble(),
			obj["z"].toDouble(),
			obj["r"].toDouble()
		);

		_graphs.insert(g->id(), g);
	}

	qInfo() << "Loading alignments...";

	QJsonArray alignments = config["alignments"].toArray();

	for ( int i = 0; i < alignments.size(); i++ )
	{
		QJsonObject obj = alignments[i].toObject();

		int id1 = obj["graphID1"].toInt();
		int id2 = obj["graphID2"].toInt();

		qInfo() << id1 << id2;

		Alignment *a = new Alignment(
			obj["edgeFile"].toString(),
			_graphs[id1],
			_graphs[id2]
		);

		_alignments.push_back(a);
	}
}



/**
 * Load ontology data from a file.
 *
 * @param filename
 */
void Database::load_ontology(const QString& filename)
{
	QFile file(filename);

	if ( !file.open(QIODevice::ReadOnly) )
	{
		qWarning("warning: unable to open ontology file");
		return;
	}

	qInfo() << "Loading ontology terms...";

	QTextStream in(&file);
	OntologyTerm ont;

	while ( !in.atEnd() )
	{
		QStringList list = in.readLine().split(" ");

		if ( list[0] == "id:" )
		{
			ont.id = list[1];
		}
		else if ( list[0] == "name:" )
		{
			list.removeFirst();
			ont.name = list.join(" ");
		}
		else if ( list[0] == "def:" )
		{
			list.removeFirst();
			ont.def = list.join(" ");

			_ontology.insert(ont.id, ont);
		}
	}

	// populate ontology terms with connected nodes
	for ( Graph *g : _graphs.values() )
	{
		const QVector<Node>& nodes = g->nodes();

		for ( int i = 0; i < nodes.size(); i++ )
		{
			for ( const QString& term : nodes[i].go_terms )
			{
				OntologyTerm& ont = _ontology[term];

				ont.connected_nodes.push_back(NodeRef {
					g->id(), i
				});
			}
		}
	}

	qInfo() << "Loaded" << _ontology.values().size() << "terms.";
}



/**
 * Print information about the database.
 */
void Database::print() const
{
	qInfo() << "Graphs:\n";
	for ( Graph *g : _graphs.values() )
	{
		g->print();
		qInfo() << "";
	}

	qInfo() << "Alignments:\n";
	for ( Alignment *a : _alignments )
	{
		a->print();
		qInfo() << "";
	}

	qInfo() << "Ontology terms:\n";
	for ( const OntologyTerm& term : _ontology.values() )
	{
		qInfo() << term.id << term.name;
		qInfo() << "";
	}
}
