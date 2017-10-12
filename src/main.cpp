#include <QApplication>
#include <QSurfaceFormat>
#include "database.h"
#include "mainwindow.h"

typedef struct {
    QString config;
    QString ont;
} args_t;

/**
 * Print command-line usage.
 */
void print_usage()
{
    fprintf(stderr,
        "Usage: ./biodep-vis [options]\n"
        "\n"
        "Options:\n"
        "  --config FILE  configuration file [config/test_M-R.json]\n"
        "  --ont FILE     ontology dictionary file [go-basic.obo]\n"
        "  --help         list help options\n"
    );
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // parse command-line arguments
    args_t args = {
        "config/test_M-R.json",
        "go-basic.obo"
    };

    QStringList options = app.arguments();

    for ( int i = 1; i < options.size(); i++ ) {
        if ( options[i] == "--config" ) {
            args.config = options[i + 1];
            i++;
        }
        else if ( options[i] == "--ont" ) {
            args.ont = options[i + 1];
            i++;
        }
        else if ( options[i] == "--help" ) {
            print_usage();
            exit(0);
        }
        else {
            print_usage();
            exit(1);
        }
    }

    // initialize OpenGL format
    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setVersion(3, 2);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

    // load data
    Database db;
    db.load_config(args.config);
    db.load_ontology(args.ont);

    // initialize window
    MainWindow window(&db);
    window.showMaximized();

    return app.exec();
}
