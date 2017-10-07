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
    int _gene_index;
    QStringList _go_terms;
    int _go_term_index;

    QListWidget *_gene_list;
    QTextEdit *_gene_desc;
    QListWidget *_go_term_list;
    QTextEdit *_go_term_desc;
    QLineEdit *_search;

public:
    MainWindow(Database *db);

public slots:
    void select_gene();
    void select_go_term();
    void search();
    void clear();

private:
    void init_controls();
    void create_gui();
    void update_gui();
};
#endif // MAINWINDOW_H
