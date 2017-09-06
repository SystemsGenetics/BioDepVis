#include "mainwindow.h"
#include <QApplication>
#include <QFile>
#include <getopt.h>
#include "graph.h"
#include "alignment.h"


std::vector <Graph> graphDatabase;
std::vector <Alignment> alignmentDatabase;
std::unordered_map<std::string, ontStruct> ontologyDB;

typedef enum {
    OPTION_ONTFILE,
    OPTION_JSONFILE,
    OPTION_HELP,
    OPTION_UNKNOWN = '?'
} option_t;


typedef struct {
    char *ont_file;
    char *json_file;
} optarg_t;

void print_usage()
{
    fprintf(stderr,
        "Usage: ./biodep-vis [options]\n"
        "\n"
        "Options:\n"
        "  --ont_file   FILE    [REQUIRED] specify the ontology text file\n"
        "  --json_file  FILE    [REQUIRED] specify the json file containing graphs and alignments\n"
        "  --help       list help options\n"
    );
}

void parser(const std::string& filename, std::vector <Graph>& gd, std::vector <Alignment>& ad, std::unordered_map<std::string, ontStruct>& ontologyDatabasePtr)
{

    QFile file(QString(filename));
    QByteArray data = file.readAll();
    QJsonObject object = QJsonDocument::fromJson(data).object();

    // read nodes
    QJsonObject nodes = object["graph"].toObject();

    for ( const QString& key : nodes.keys() ) {
        QJsonObject jsonNode = nodes[key].toObject();

        node_t node = {
            jsonNode["id"].toInt(),
            jsonNode["name"].toString(),
            jsonNode["taxid"].toInt(),
            jsonNode["fileLocation"].toString(),
            jsonNode["clusterLocation"].toString(),
            jsonNode["Ontology"].toString(),
            jsonNode["x"].toInt(),
            jsonNode["y"].toInt(),
            jsonNode["z"].toInt(),
            jsonNode["w"].toInt(),
            jsonNode["h"].toInt()
        };
        Graph g(node.id, node.name, node.filename, node.filenamecluster,node.ontFile,node.x,node.y,node.z,node.w,node.h,ontologyDatabasePtr);
        gd.push_back(g);
   }

    //read edges
    QJsonObject edges = object["alignment"].toObject();

    for(const QString& key : edges.keys()){
        QJsonObject jsonEdge = edges[key].toObject();

        edge_t edge ={
            jsonEdge["graphID1"].toInt(),
            jsonEdge["graphID2"].toInt(),
            jsonEdge["filelocation"].toString()//fix format
        };

        Alignment a(edge.graphID1,edge.graphID2,edge.filelocation);
        ad.push_back(a);

    }

}

int main(int argc, char *argv[])
{
    optarg_t args = {
        NULL,
        NULL
    };

    struct option long_options[] = {
        { "ont_file", required_argument, 0, OPTION_ONTFILE },
        { "json_file", required_argument, 0, OPTION_JSONFILE },
        { "help", no_argument, 0, OPTION_HELP },
        { 0, 0, 0, 0 }
    };

    int opt;
    while ( (opt = getopt_long_only(argc, argv, "", long_options, NULL)) != -1 ) {
        switch ( opt ) {
        case OPTION_ONTFILE:
            args.ont_file = optarg;
            break;
        case OPTION_JSONFILE:
            args.json_file = optarg;
            break;
        case OPTION_HELP:
            print_usage();
            exit(1);
        case OPTION_UNKNOWN:
            print_usage();
            exit(1);
        }
    }
    if(args.ont_file == NULL){
        args.ont_file = "ont_data.txt";
    }
    if(args.json_file == NULL){
        args.json_file = "input.json";
    }

    // validate arguments
    if ( !args.ont_file || !args.json_file ) {
        print_usage();
        exit(1);
    }

    parser(args.json_file, graphDatabase, alignmentDatabase, ontologyDB);

    // read the ontology file
    //readOntFile(&ontologyDB, args.ont_file);

    QApplication a(argc, argv);

    MainWindow w;
    w.resize(1280, 720);
    w.show();

    return a.exec();
}


