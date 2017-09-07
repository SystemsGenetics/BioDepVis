#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include "database.h"

/**
 * Load graph and alignment data from a configuration file.
 *
 * @param filename
 */
bool Database::load_alignments(const QString& filename)
{
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    QByteArray data = file.readAll();
    QJsonObject object = QJsonDocument::fromJson(data).object();

    // read graphs
    QJsonObject nodes = object["graph"].toObject();

    for ( const QString& key : nodes.keys() ) {
        QJsonObject jsonNode = nodes[key].toObject();

        Graph g(
            jsonNode["id"].toInt(),
            jsonNode["name"].toString(),
            jsonNode["fileLocation"].toString(),
            jsonNode["clusterLocation"].toString(),
            jsonNode["Ontology"].toString(),
            jsonNode["x"].toInt(),
            jsonNode["y"].toInt(),
            jsonNode["z"].toInt(),
            jsonNode["w"].toInt(),
            jsonNode["h"].toInt()
        );

        this->_graphs.push_back(g);
   }

    // read alignments
    QJsonObject edges = object["alignment"].toObject();

    for ( const QString& key : edges.keys() ) {
        QJsonObject jsonEdge = edges[key].toObject();

        Alignment a(
			jsonEdge["filelocation"].toString(),
			jsonEdge["graphID1"].toInt(),
			jsonEdge["graphID2"].toInt()
        );

        this->_alignments.push_back(a);
    }

    return true;
}

/**
 * Load ontology data from a file.
 *
 * @param filename
 */
bool Database::load_ontology(const QString& filename)
{
	// TODO: stub

    return false;
}
