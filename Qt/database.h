#ifndef DATABASE_H
#define DATABASE_H

#include <string>
#include <unordered_map>
#include <vector>
#include <QHash>
#include "alignment.h"


typedef struct {
    int nodeSelected;
    int graphSelected;
} nodeSelectedStruct;

typedef struct {
    QString id;
    QString name;
    QString def;
    int index;
    std::vector<nodeSelectedStruct> connectedNodes;
} ont_term_t;

class Database {
public:
    QHash<int,Graph*> _graphs;
    std::vector<Alignment> _alignments;
    QHash<QString, ont_term_t> _ontology;

public:
    Database(){};

    bool load_config(const QString& filename);
    bool load_ontology(const QString& filename);
};

#endif
