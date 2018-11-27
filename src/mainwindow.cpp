#include "glwidget.h"
#include "mainwindow.h"



MainWindow::MainWindow(Database *db):
	_db(db)
{
	init_controls();
	create_gui();
}



void MainWindow::init_controls()
{
	_genes.clear();
	_gene_index = -1;
	_go_terms.clear();
	_go_term_index = -1;
}



void MainWindow::create_gui()
{
	QGridLayout *layout = new QGridLayout;

	// search interface
	QGroupBox *searchGroup = new QGroupBox("Gene Ontology Search");
	QVBoxLayout *searchLayout = new QVBoxLayout;
	searchGroup->setLayout(searchLayout);

	QLabel *geneListLabel = new QLabel("Genes");
	_gene_list = new QListWidget();
	connect(_gene_list, SIGNAL(itemActivated(QListWidgetItem *)), this, SLOT(selectGene()));

	QLabel *geneDescLabel = new QLabel("Description");
	_gene_desc = new QTextEdit();
	_gene_desc->setReadOnly(true);

	QFrame *line1 = new QFrame;
	line1->setFrameShape(QFrame::HLine);
	line1->setFrameShadow(QFrame::Sunken);

	QLabel *goTermListLabel = new QLabel("Ontology Terms");
	_go_term_list = new QListWidget();
	connect(_go_term_list, SIGNAL(itemActivated(QListWidgetItem *)), this, SLOT(selectGoTerm()));

	QLabel *goTermDescLabel = new QLabel("Description");
	_go_term_desc = new QTextEdit();
	_go_term_desc->setReadOnly(true);

	QFrame *line2 = new QFrame;
	line2->setFrameShape(QFrame::HLine);
	line2->setFrameShadow(QFrame::Sunken);

	QLabel *searchTermLabel = new QLabel("Search Term");
	_search = new QLineEdit();

	QPushButton *searchButton = new QPushButton("Search");
	connect(searchButton, SIGNAL(clicked()), this, SLOT(search()));

	QPushButton *clearButton = new QPushButton("Clear");
	connect(clearButton, SIGNAL(clicked()), this, SLOT(clear()));

	QPushButton *extractButton = new QPushButton("Extract Subgraphs");
	connect(extractButton, SIGNAL(clicked()), this, SLOT(extractSubgraphs()));

	searchLayout->addWidget(geneListLabel);
	searchLayout->addWidget(_gene_list);
	searchLayout->addWidget(geneDescLabel);
	searchLayout->addWidget(_gene_desc);
	searchLayout->addWidget(line1);
	searchLayout->addWidget(goTermListLabel);
	searchLayout->addWidget(_go_term_list);
	searchLayout->addWidget(goTermDescLabel);
	searchLayout->addWidget(_go_term_desc);
	searchLayout->addWidget(line2);
	searchLayout->addWidget(searchTermLabel);
	searchLayout->addWidget(_search);
	searchLayout->addWidget(searchButton);
	searchLayout->addWidget(clearButton);
	searchLayout->addWidget(extractButton);

	// visualizer
	QGroupBox *visGroup = new QGroupBox("Visualizer");
	QVBoxLayout *visLayout = new QVBoxLayout;
	visGroup->setLayout(visLayout);

	GLWidget *glWidget = new GLWidget(_db);
	connect(glWidget, SIGNAL(nodesSelected(const QVector<node_ref_t>&)), this, SLOT(setSelectedGenes(const QVector<node_ref_t>&)));
	connect(this, SIGNAL(genesSelected(const QVector<node_ref_t>&)), glWidget, SLOT(setSelectedNodes(const QVector<node_ref_t>&)));

	visLayout->addWidget(glWidget);

	// keyboard legend
	QGroupBox *legendGroup = new QGroupBox("Keyboard Controls");
	QFormLayout *legendLayout = new QFormLayout;
	legendGroup->setLayout(legendLayout);

	QVector<QPair<QString, QString>> controls {
		{ "R", "Reset View" },
		{ "W/S", "Pan Up/Down" },
		{ "A/D", "Pan Left/Right" },
		{ "Q/E", "Zoom" },
		{ "I/K", "Rotate Up/Down" },
		{ "J/L", "Rotate Left/Right" },
		{ "U/O", "Rotate Z-axis" },
		{ "G", "Toggle GPU" },
		{ "Space", "Toggle force-directed layout" },
		{ "Z", "Toggle graph" },
		{ "X", "Toggle alignment" },
		{ "C", "Toggle module coloring" },
		{ "B", "Toggle multi-select" }
	};

	for ( auto& ctrl : controls )
	{
		QLabel *label1 = new QLabel(ctrl.first);
		QLabel *label2 = new QLabel(ctrl.second);
		legendLayout->addRow(label1, label2);
	}

	// add groups to layout
	layout->setColumnStretch(0, 1);
	layout->setColumnStretch(1, 3);
	layout->setColumnStretch(2, 1);
	layout->addWidget(searchGroup, 0, 0);
	layout->addWidget(visGroup,    0, 1);
	layout->addWidget(legendGroup, 0, 2);
	setLayout(layout);

	// set focus
	glWidget->setFocus();
}



void MainWindow::update_gui()
{
	if ( _gene_index == -1 )
	{
		// update gene list
		_gene_list->clear();

		for ( const node_ref_t& ref : _genes )
		{
			_gene_list->addItem(_db->node(ref).name);
		}

		_gene_desc->setText("");
	}
	else
	{
		// update gene description
		const node_ref_t& ref = _genes[_gene_index];

		_gene_desc->setText(_db->node(ref).name);
	}

	if ( _go_term_index == -1 )
	{
		// update ontology term list
		_go_term_list->clear();

		for ( const QString& id : _go_terms )
		{
			_go_term_list->addItem(id);
		}

		_go_term_desc->setText("");
	}
	else
	{
		// update ontology term description
		const QString& id = _go_terms[_go_term_index];

		_go_term_desc->setText(_db->ontology()[id].def);
	}
}



void MainWindow::setSelectedGenes(const QVector<node_ref_t>& genes)
{
	init_controls();

	_genes = genes;

	emit genesSelected(_genes);
	update_gui();
}



void MainWindow::selectGene()
{
	_gene_index = 0;
	while ( !_gene_list->item(_gene_index)->isSelected() )
	{
		_gene_index++;
	}

	// update go term list
	const node_ref_t& ref = _genes[_gene_index];

	_go_terms = _db->node(ref).go_terms;
	_go_term_index = -1;

	update_gui();
}



void MainWindow::selectGoTerm()
{
	_go_term_index = 0;
	while ( !_go_term_list->item(_go_term_index)->isSelected() )
	{
		_go_term_index++;
	}

	update_gui();
}



void MainWindow::search()
{
	QString term = _search->text();

	if ( term.isEmpty() )
	{
		return;
	}

	init_controls();

	for ( const ont_term_t& ont : _db->ontology().values() )
	{
		if ( ont.def.contains(term) )
		{
			_genes.append(ont.connected_nodes);
		}
	}

	emit genesSelected(_genes);
	update_gui();
}



void MainWindow::clear()
{
	init_controls();

	emit genesSelected(_genes);
	update_gui();
}



void MainWindow::extractSubgraphs()
{
	for ( Alignment *a : _db->alignments() )
	{
		a->extract_subgraphs();
	}
}
