#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets>
#include "database.h"

class MainWindow : public QWidget
{
    Q_OBJECT

private:
    Database *_db;
    QVector<node_ref_t> _genes;

    QListWidget *_gene_list;
    QLabel *_gene_desc;
    QListWidget *_go_term_list;
    QLabel *_go_term_desc;
    QLineEdit *_search;

public:
    MainWindow(Database *db);

public slots:
    void search();
    void clear();

private:
    void create_gui();
    void update_gui();
};
#endif // MAINWINDOW_H
