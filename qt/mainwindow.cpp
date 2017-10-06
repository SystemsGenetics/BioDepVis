#include "mainwindow.h"
#include "glwidget.h"

MainWindow::MainWindow(Database *db)
{
    this->_db = db;

    QGridLayout *layout = new QGridLayout;

    // search interface
    QGroupBox *searchGroup = new QGroupBox("Search");
    QGridLayout *searchLayout = new QGridLayout;
    searchGroup->setLayout(searchLayout);

    QLabel *geneListLabel = new QLabel("Genes");
    this->_gene_list = new QListWidget();

    QLabel *geneDescLabel = new QLabel("Description");
    this->_gene_desc = new QLabel();
    QScrollArea *geneDescScrollArea = new QScrollArea();
    geneDescScrollArea->setWidget(this->_gene_desc);

    QLabel *goTermListLabel = new QLabel("Ontology Terms");
    this->_go_term_list = new QListWidget();

    QLabel *goTermDescLabel = new QLabel("Description");
    this->_go_term_desc = new QLabel();
    QScrollArea *goTermDescScrollArea = new QScrollArea();
    goTermDescScrollArea->setWidget(this->_go_term_desc);

    QLabel *searchTermLabel = new QLabel("Search Term");
    this->_search = new QLineEdit();

    QPushButton *searchButton = new QPushButton("Search");
    connect(searchButton, SIGNAL(clicked()), this, SLOT(search()));

    QPushButton *clearButton = new QPushButton("Clear");
    connect(clearButton, SIGNAL(clicked()), this, SLOT(clear()));

    searchLayout->addWidget(geneListLabel,         0, 0, 1, 1);
    searchLayout->addWidget(this->_gene_list,      1, 0, 1, 1);
    searchLayout->addWidget(geneDescLabel,         2, 0, 1, 1);
    searchLayout->addWidget(geneDescScrollArea,    3, 0, 1, 1);
    searchLayout->addWidget(goTermListLabel,       4, 0, 1, 1);
    searchLayout->addWidget(this->_go_term_list,   5, 0, 1, 1);
    searchLayout->addWidget(goTermDescLabel,       6, 0, 1, 1);
    searchLayout->addWidget(goTermDescScrollArea,  7, 0, 1, 1);
    searchLayout->addWidget(searchTermLabel,       8, 0, 1, 1);
    searchLayout->addWidget(this->_search,         9, 0, 1, 1);
    searchLayout->addWidget(searchButton,         10, 0, 1, 1);
    searchLayout->addWidget(clearButton,          11, 0, 1, 1);

    // visualizer
    QGroupBox *visGroup = new QGroupBox("Visualizer");
    QVBoxLayout *visLayout = new QVBoxLayout;
    visGroup->setLayout(visLayout);

    GLWidget *glWidget = new GLWidget(this->_db);

    visLayout->addWidget(glWidget);

    // keyboard legend
    QGroupBox *legendGroup = new QGroupBox("Controls");
    QGridLayout *legendLayout = new QGridLayout;
    legendGroup->setLayout(legendLayout);

    QVector<QPair<QString, QString>> controls {
        { "R", "Reset View" },
        { "G", "Toggle GPU" },
        { "Space", "Toggle FDL" },
        { ",", "Toggle module coloring" },
        { "V", "Toggle alignment" },
        { "W/S", "Rotate X-axis" },
        { "A/D", "Rotate Y-axis" },
        { "Q/E", "Zoom" },
        { "I/K", "Pan Left/Right" },
        { "J/L", "Pan Up/Down" }
    };

    legendLayout->setColumnStretch(0, 1);
    legendLayout->setColumnStretch(1, 3);

    for ( int i = 0; i < controls.size(); i++ ) {
        auto& ctrl = controls[i];

        QLabel *label1 = new QLabel(ctrl.first);
        QLabel *label2 = new QLabel(ctrl.second);
        legendLayout->addWidget(label1, i, 0, 1, 1);
        legendLayout->addWidget(label2, i, 1, 1, 1);
    }

    // add groups to layout
    layout->setColumnStretch(0, 1);
    layout->setColumnStretch(1, 3);
    layout->setColumnStretch(2, 1);
    layout->addWidget(searchGroup, 0, 0);
    layout->addWidget(visGroup,    0, 1);
    layout->addWidget(legendGroup, 0, 2);
    this->setLayout(layout);
}

void MainWindow::search()
{
    QString term = this->_search->text();

    this->_genes.clear();

    for ( const ont_term_t& ont : this->_db->ontology().values() ) {
        if ( ont.def.contains(term) ) {
            this->_genes.append(ont.connected_nodes);
        }
    }

    this->update_gui();
}

void MainWindow::clear()
{
    this->_genes.clear();

    this->update_gui();
}

void MainWindow::update_gui()
{
    // update gene list
    this->_gene_list->clear();

    for ( const node_ref_t& ref : this->_genes ) {
        const graph_node_t& node = this->_db->graphs()[ref.graph_id]->nodes()[ref.node_id];

        this->_gene_list->addItem(node.name);
    }

    // update gene description
    // TODO

    // update ontology term list
    // TODO

    // update ontology term description
    // TODO
}
