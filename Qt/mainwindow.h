#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>

class MainWindow : public QWidget
{
    Q_OBJECT

private:
    struct {
        QLineEdit *filename;
    } _treeForm;

public:
    MainWindow();

public slots:
    bool search();
    bool clear();

private:
    void createGUI();

};
#endif // MAINWINDOW_H
