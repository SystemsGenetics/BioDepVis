#include "alignment.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>

Alignment::Alignment(const QString& filename, Graph *graph1, Graph *graph2)
{
    this->_graph1 = graph1;
    this->_graph2 = graph2;

    load_alignment(filename);
}

bool Alignment::load_alignment(const QString& filename)
{
    QFile file(filename);

    if ( !file.open(QIODevice::ReadOnly) ) {
        return false;
    }

    QTextStream in(&file);

    while ( !in.atEnd() ) {
        QStringList list = in.readLine().split("\t");
        QString node1 = list[0];
        QString node2 = list[1];

        // TODO
    }

    return true;
}

void Alignment::print() const
{
    qDebug() << this->_graph1->id() << this->_graph2->id();
}
