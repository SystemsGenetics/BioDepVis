#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets>
#include "database.h"



class MainWindow : public QWidget
{
	Q_OBJECT

private:
	Database *_db;
	QVector<NodeRef> _genes;
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
	void setSelectedGenes(const QVector<NodeRef>& genes);
	void selectGene();
	void selectGoTerm();
	void search();
	void clear();
	void extractSubgraphs();

signals:
	void genesSelected(const QVector<NodeRef>& genes);

private:
	void init_controls();
	void create_gui();
	void update_gui();
};



#endif // MAINWINDOW_H
