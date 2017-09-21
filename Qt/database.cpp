#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTextStream>
#include "database.h"
#include <QDebug>

/**
 * Load graph and alignment data from a configuration file.
 *
 * @param filename
 */
bool Database::load_config(const QString& filename)
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

        Graph *g = new Graph(
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

        this->_graphs.insert(g->id(), g);
   }

    // read alignments
    QJsonObject edges = object["alignment"].toObject();

    for ( const QString& key : edges.keys() ) {
        QJsonObject jsonEdge = edges[key].toObject();

        int id1 = jsonEdge["graphID1"].toInt();
        int id2 = jsonEdge["graphID2"].toInt();

        // search graphs for ids
        Graph* g1 = _graphs.value(id1);
        Graph* g2 = _graphs.value(id2);

        // construct alignment
        Alignment a(
            jsonEdge["filelocation"].toString(),
            g1,g2
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
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QString line = in.readLine();
        QStringList list = line.split(" ");

        ont_term_t ont;
        if ( list[0] == "id:" ) {
            ont.id = list[1];
        }
        else if ( list[0] == "name:" ) {
            list.removeFirst();
            ont.name = list.join(" ");
        }
        else if ( list[0] == "def:" ) {
            list.removeFirst();
            ont.def = list.join(" ");

            this->_ontology[ont.id] = ont;
        }
    }

    // populate ontology terms with connected nodes
    for ( Graph *g : this->_graphs.values() ) {
        QVector<node_t> nodes = g->node_list();

        for ( int i = 0; i < nodes.size(); i++ ) {
            for ( const QString& term : nodes[i].go_terms ) {
                ont_term_t& ont = this->_ontology[term];

                ont.connectedNodes.push_back(nodeSelectedStruct {
                    i, g->id() - 1
                });
            }
        }
    }

    return true;
}

void Database::print() const
{
    qDebug() << "Ontology terms:\n";
    for ( const ont_term_t& term : this->_ontology.values() ) {
        qDebug() << term.id << term.name;
        qDebug() << "";
    }

    qDebug() << "Graphs:\n";
    for ( Graph *g : this->_graphs.values() ) {
        g->print();
        qDebug() << "";
    }

    qDebug() << "Alignments:\n";
    for ( const Alignment& a : this->_alignments ) {
        a.print();
        qDebug() << "";
    }
}
