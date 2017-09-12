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


    QString line;
    QFile file(filename);
    QTextStream in(&file);

    if (!file.open(QIODevice::ReadOnly) ) {

        return false;
    }
    qDebug("Test1");

    ont_term_t ont;

    while(!in.atEnd()){

        line = in.readLine();
        QStringList list = line.split(" ");

        if(list[0] == "id:"){
            ont.id = list[1];
        }
        if(list[0] == "name:"){
            list.removeFirst();
            ont.name = list.join(" ");
        }
        if(list[0] == "def:"){
            list.removeFirst();
            ont.def = list.join(" ");
            _ontology[ont.id] = ont;
        }
    }

    qDebug("Test2");

    for( ont_term_t& term : _ontology){
        qDebug("Test3");
        //qDebug() << term.id << term.name << term.def << "\n";
    }

    return false;
}
