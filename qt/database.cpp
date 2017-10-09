#include <QDebug>
#include <QFile>
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

    if ( !file.open(QIODevice::ReadOnly) ) {
        qWarning("warning: unable to open config file");
        return;
    }

    QByteArray data = file.readAll();
    QJsonObject object = QJsonDocument::fromJson(data).object();

    qInfo() << "Loading graphs...";

    QJsonObject graphs = object["graph"].toObject();

    for ( const QString& key : graphs.keys() ) {
        QJsonObject obj = graphs[key].toObject();

        qInfo() << obj["id"].toInt() << obj["name"].toString();

        Graph *g = new Graph(
            obj["id"].toInt(),
            obj["name"].toString(),
            obj["clusterLocation"].toString(),
            obj["fileLocation"].toString(),
            obj["Ontology"].toString(),
            obj["x"].toDouble(),
            obj["y"].toDouble(),
            obj["z"].toDouble(),
            obj["w"].toDouble(),
            obj["h"].toDouble()
        );

        qInfo() << "Loaded" << g->nodes().size() << "nodes," << g->edges().size() << "edges.";

        this->_graphs.insert(g->id(), g);
    }

    qInfo() << "Loading alignments...";

    QJsonObject alignments = object["alignment"].toObject();

    for ( const QString& key : alignments.keys() ) {
        QJsonObject obj = alignments[key].toObject();

        int id1 = obj["graphID1"].toInt();
        int id2 = obj["graphID2"].toInt();

        qInfo() << id1 << id2;

        Alignment *a = new Alignment(
            obj["filelocation"].toString(),
            this->_graphs[id1],
            this->_graphs[id2]
        );

        this->_alignments.push_back(a);
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

    if ( !file.open(QIODevice::ReadOnly) ) {
        qWarning("warning: unable to open ontology file");
        return;
    }

    qInfo() << "Loading ontology terms...";

    QTextStream in(&file);
    ont_term_t ont;

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split(" ");

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

            this->_ontology.insert(ont.id, ont);
        }
    }

    // populate ontology terms with connected nodes
    for ( Graph *g : this->_graphs.values() ) {
        const QVector<graph_node_t>& nodes = g->nodes();

        for ( int i = 0; i < nodes.size(); i++ ) {
            for ( const QString& term : nodes[i].go_terms ) {
                ont_term_t& ont = this->_ontology[term];

                ont.connected_nodes.push_back(node_ref_t {
                    g->id(), i
                });
            }
        }
    }

    qInfo() << "Loaded" << this->_ontology.values().size() << "terms.";
}

void Database::print() const
{
    qInfo() << "Graphs:\n";
    for ( Graph *g : this->_graphs.values() ) {
        g->print();
        qInfo() << "";
    }

    qInfo() << "Alignments:\n";
    for ( Alignment *a : this->_alignments ) {
        a->print();
        qInfo() << "";
    }

    qInfo() << "Ontology terms:\n";
    for ( const ont_term_t& term : this->_ontology.values() ) {
        qInfo() << term.id << term.name;
        qInfo() << "";
    }
}
