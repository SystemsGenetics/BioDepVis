#ifndef DATABASE_H
#define DATABASE_H

#include <QHash>
#include <QMap>
#include "alignment.h"

typedef struct {
    int graph_id;
    int node_id;
} node_ref_t;

typedef struct {
    QString id;
    QString name;
    QString def;
    QVector<node_ref_t> connected_nodes;
} ont_term_t;

class Database {
private:
    QMap<int, Graph *> _graphs;
    QVector<Alignment *> _alignments;
    QHash<QString, ont_term_t> _ontology;

public:
    Database() {};

    QMap<int, Graph *>& graphs() { return this->_graphs; }
    QVector<Alignment *>& alignments() { return this->_alignments; }
    QHash<QString, ont_term_t> ontology() { return this->_ontology; }

    void load_config(const QString& filename);
    void load_ontology(const QString& filename);

    void print() const;
};

#endif
