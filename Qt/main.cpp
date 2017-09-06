#include "mainwindow.h"
#include <QApplication>
#include <getopt.h>

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

    // read the ontology file
    //readOntFile(&ontologyDB, args.ont_file);

    QApplication a(argc, argv);

    MainWindow w;
    w.resize(1280, 720);
    w.show();

    return a.exec();
}
