#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalMapper>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QScrollArea>
#include "mainwindow.h"
#include "visualizer.h"

MainWindow::MainWindow(Database *db)
{
    QGridLayout *layout = new QGridLayout;

    // Ontology Window
    QGroupBox *OntologyGroup = new QGroupBox("Ontology Window");

    QLabel *resultsLabel = new QLabel("Selected Results");
    QLabel *descriptionLabel = new QLabel("Description");
    QWidget *resultsContent = new QWidget();

    QWidget *descriptionContent = new QWidget();

    QScrollArea *resultsScrollArea = new QScrollArea();
    resultsScrollArea->setWidget(resultsContent);

    QScrollArea *descriptionScrollArea = new QScrollArea();
    descriptionScrollArea->setWidget(descriptionContent);

    QGridLayout *OntologyLayout = new QGridLayout;
    OntologyLayout->addWidget(resultsLabel, 0, 0);
    OntologyLayout->addWidget(resultsScrollArea,1,0);
    OntologyLayout->addWidget(descriptionLabel,2,0);
    OntologyLayout->addWidget(descriptionScrollArea,3,0);
    OntologyGroup->setLayout(OntologyLayout);

    // Go Term List
    QGroupBox *GoTermGroup = new QGroupBox("Go Term List");

    QWidget *termsContent = new QWidget();

    QWidget *termDescriptionContent = new QWidget();

    QScrollArea *termsScrollArea = new QScrollArea();
    termsScrollArea->setWidget(termsContent);

    QScrollArea *termDescriptionScrollArea = new QScrollArea();
    termDescriptionScrollArea->setWidget(termDescriptionContent);

    QGridLayout *GoTermLayout = new QGridLayout;
    GoTermLayout->addWidget(termsScrollArea,1,0);
    GoTermLayout->addWidget(termDescriptionScrollArea,2,0);
    GoTermGroup->setLayout(GoTermLayout);

    // Search Window
    QGroupBox *SearchWindowGroup = new QGroupBox("Search Window");
    QLabel *searchTermLabel = new QLabel("Search Term");

    QWidget *searchTermsContent = new QWidget();
    QScrollArea *searchTermsScrollArea = new QScrollArea();
    searchTermsScrollArea->setWidget(searchTermsContent);

    QPushButton *searchButton = new QPushButton("Search");
    connect(searchButton, SIGNAL(clicked()), this, SLOT(search()));

    QPushButton *clearButton = new QPushButton("Clear");
    connect(clearButton, SIGNAL(clicked()), this, SLOT(clear()));

    QGridLayout *SearchWindowLayout = new QGridLayout;
    SearchWindowLayout->addWidget(searchTermLabel, 0, 0);
    SearchWindowLayout->addWidget(searchTermsScrollArea,1,0);
    SearchWindowLayout->addWidget(searchButton,2,0);
    SearchWindowLayout->addWidget(clearButton,3,0);
    SearchWindowGroup->setLayout(SearchWindowLayout);

    // Visualization
    QGroupBox *VisualizationGroup = new QGroupBox;

    Visualizer *visualizer = new Visualizer(db);

    QVBoxLayout *VisualizationLayout = new QVBoxLayout;
    VisualizationLayout->addWidget(visualizer);
    VisualizationGroup->setLayout(VisualizationLayout);

    // Legend
    QGroupBox *LegendGroup = new QGroupBox("Controls");
    QVBoxLayout *LegendLayout = new QVBoxLayout;

    QStringList controlHeaders {
        "Key",
        "Action"
    };

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

    QTableWidget *controlTable = new QTableWidget(controls.size(), 2);
    controlTable->setHorizontalHeaderLabels(controlHeaders);
    controlTable->horizontalHeader()->setStretchLastSection(true);
    controlTable->verticalHeader()->setVisible(false);

    for ( int i = 0; i < controls.size(); i++ ) {
        auto& ctrl = controls[i];

        controlTable->setItem(i, 0, new QTableWidgetItem(ctrl.first));
        controlTable->setItem(i, 1, new QTableWidgetItem(ctrl.second));
    }

    LegendLayout->addWidget(controlTable, 0, 0);
    LegendGroup->setLayout(LegendLayout);

    // Add Groups to Layout
    layout->setColumnStretch(0, 1);
    layout->setColumnStretch(1, 2);
    layout->setColumnStretch(2, 2);
    layout->addWidget(OntologyGroup,      0, 0, 2, 1);
    layout->addWidget(GoTermGroup,        2, 0, 1, 1);
    layout->addWidget(SearchWindowGroup,  3, 0, 1, 1);
    layout->addWidget(VisualizationGroup, 0, 1, 4, 3);
    layout->addWidget(LegendGroup,        0, 4, 4, 3);
    this->setLayout(layout);
}

bool MainWindow::search()
{
    // TODO: stub

    return true;
}
bool MainWindow::clear()
{
    // TODO: stub

    return true;
}
