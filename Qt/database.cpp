#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDebug>
#include <QTextStream>
#include "database.h"

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
    QJsonObject graphs = object["graph"].toObject();

    for ( const QString& key : graphs.keys() ) {
        QJsonObject obj = graphs[key].toObject();

        Graph *g = new Graph(
            obj["id"].toInt(),
            obj["name"].toString(),
            obj["fileLocation"].toString(),
            obj["clusterLocation"].toString(),
            obj["Ontology"].toString(),
            obj["x"].toInt(),
            obj["y"].toInt(),
            obj["z"].toInt(),
            obj["w"].toInt(),
            obj["h"].toInt()
        );

        this->_graphs.insert(g->id(), g);
   }

    // read alignments
    QJsonObject alignments = object["alignment"].toObject();

    for ( const QString& key : alignments.keys() ) {
        QJsonObject obj = alignments[key].toObject();

        int id1 = obj["graphID1"].toInt();
        int id2 = obj["graphID2"].toInt();

        Alignment a(
            obj["filelocation"].toString(),
            this->_graphs[id1],
            this->_graphs[id2]
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
        QStringList list = in.readLine().split(" ");

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
        QVector<node_t> nodes = g->nodes();

        for ( int i = 0; i < nodes.size(); i++ ) {
            for ( const QString& term : nodes[i].go_terms ) {
                ont_term_t& ont = this->_ontology[term];

                ont.connected_nodes.push_back(node_ref_t {
                    g->id() - 1, i
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
