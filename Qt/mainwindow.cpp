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

    //Go Term List
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

    //Search Window
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


    //Visualization
    QGroupBox *VisualizationGroup = new QGroupBox;

    Visualizer *visualizer = new Visualizer(db);

    QVBoxLayout *VisualizationLayout = new QVBoxLayout;
    VisualizationLayout->addWidget(visualizer);
    VisualizationGroup->setLayout(VisualizationLayout);

    //Legend
    QGroupBox *LegendGroup = new QGroupBox("Controls");
    QVBoxLayout *LegendLayout = new QVBoxLayout;

    QStringList controlHeaders {
        "Key",
        "Action"
    };

    QTableWidget * controlTable = new QTableWidget(21,2);
    controlTable->setHorizontalHeaderLabels(controlHeaders);
    controlTable->horizontalHeader()->setStretchLastSection(true);
    controlTable->verticalHeader()->setVisible(false);

    //Control Commands
    controlTable->setItem(0, 0, new QTableWidgetItem("Space"));
    controlTable->setItem(0, 1, new QTableWidgetItem("FDL"));

    controlTable->setItem(1, 0, new QTableWidgetItem("Q"));
    controlTable->setItem(1, 1, new QTableWidgetItem("Zoom Out"));

    controlTable->setItem(2, 0, new QTableWidgetItem("W"));
    controlTable->setItem(2, 1, new QTableWidgetItem("Rotate on y axis"));

    controlTable->setItem(3, 0, new QTableWidgetItem("E"));
    controlTable->setItem(3, 1, new QTableWidgetItem("Zoom In"));

    controlTable->setItem(4, 0, new QTableWidgetItem("R"));
    controlTable->setItem(4, 1, new QTableWidgetItem("Reset View"));

    controlTable->setItem(5, 0, new QTableWidgetItem("U,A,J"));
    controlTable->setItem(5, 1, new QTableWidgetItem("Pan Down View"));

    controlTable->setItem(6, 0, new QTableWidgetItem("O"));
    controlTable->setItem(6, 1, new QTableWidgetItem("Pan Down View (slow)"));

    controlTable->setItem(7, 0, new QTableWidgetItem("I"));
    controlTable->setItem(7, 1, new QTableWidgetItem("Pan Left View"));

    controlTable->setItem(8, 0, new QTableWidgetItem("S,K"));
    controlTable->setItem(8, 1, new QTableWidgetItem("Pan Right View"));

    controlTable->setItem(9, 0, new QTableWidgetItem("D"));
    controlTable->setItem(9, 1, new QTableWidgetItem("Rotate on X axis and zoom out"));

    controlTable->setItem(10, 0, new QTableWidgetItem("L"));
    controlTable->setItem(10, 1, new QTableWidgetItem(" Pan Up View"));

    controlTable->setItem(11, 0, new QTableWidgetItem("X"));
    controlTable->setItem(11, 1, new QTableWidgetItem("Show only selected nodes"));

    controlTable->setItem(12, 0, new QTableWidgetItem("V"));
    controlTable->setItem(12, 1, new QTableWidgetItem("Change Edge Design (curved to 2D)"));

    controlTable->setItem(13, 0, new QTableWidgetItem(","));
    controlTable->setItem(13, 1, new QTableWidgetItem("Show node type"));

    LegendLayout->addWidget(controlTable,0,0);
    LegendGroup->setLayout(LegendLayout);

    //Add Groups to Layout
    layout->setColumnStretch(0, 1);
    layout->setColumnStretch(1, 2);
    layout->setColumnStretch(2, 2);
    layout->addWidget(OntologyGroup, 0, 0,2,1);
    layout->addWidget(GoTermGroup, 2, 0,1,1);
    layout->addWidget(SearchWindowGroup, 3, 0,1,1);
    layout->addWidget(VisualizationGroup, 0, 1, 4, 3);
    layout->addWidget(LegendGroup, 0, 4, 4, 3);
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
