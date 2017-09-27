#include <getopt.h>
#include <QApplication>
#include <QSurfaceFormat>
#include "database.h"
#include "mainwindow.h"

typedef enum {
    OPTION_ONTFILE,
    OPTION_JSONFILE,
    OPTION_HELP,
    OPTION_UNKNOWN = '?'
} option_t;

typedef struct {
    const char *ont_file;
    const char *json_file;
} optarg_t;

/**
 * Print command-line usage.
 */
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
    // parse command-line arguments
    optarg_t args = {
        "go-basic.obo",
        "input.json"
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

    // validate arguments
    if ( !args.ont_file || !args.json_file ) {
        print_usage();
        exit(1);
    }

    // load data
    Database db;

    db.load_config(args.json_file);
    db.load_ontology(args.ont_file);

    // start application
    QApplication a(argc, argv);

    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setVersion(3, 2);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

    MainWindow w;
    w.resize(1280, 720);
    w.show();

    return a.exec();
}
