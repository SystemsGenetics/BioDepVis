#include "alignment.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>

Alignment::Alignment(const QString& filename, Graph* graph1, Graph* graph2)
{
    this->graph1 = graph1;
    this->graph2 = graph2;
    load_alignment(filename);
}

bool Alignment::load_alignment(const QString& filename)
{
    QFile file(filename);
    QTextStream in(&file);
    QString node1;
    QString node2;
    QString line;

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    while(!in.atEnd()){
        line = in.readLine();
        QStringList list = line.split("\t");
        node1 = list[0];
        node2 = list[1];

        //Todo:
        //check in nodeListMap
    }

    return true;
}

void Alignment::print() const
{
    qDebug() << graph1->id() << graph2->id();
}
