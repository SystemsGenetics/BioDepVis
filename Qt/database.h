#ifndef DATABASE_H
#define DATABASE_H

#include <QHash>
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
    QHash<int, Graph *> _graphs;
    QVector<Alignment> _alignments;
    QHash<QString, ont_term_t> _ontology;

public:
    Database() {};

    QHash<int, Graph *>& graphs() { return this->_graphs; }
    QVector<Alignment>& alignments() { return this->_alignments; }

    void load_config(const QString& filename);
    void load_ontology(const QString& filename);

    void print() const;
};

#endif
