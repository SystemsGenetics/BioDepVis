#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QLineEdit>
#include <QMainWindow>
#include "database.h"

class MainWindow : public QWidget
{
    Q_OBJECT

public:
    MainWindow(Database *db);

public slots:
    bool search();
    bool clear();
};
#endif // MAINWINDOW_H
