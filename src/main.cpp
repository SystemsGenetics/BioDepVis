#include <QApplication>
#include <QSurfaceFormat>
#include "database.h"
#include "fdl.h"
#include "mainwindow.h"



typedef struct
{
	QString config;
	QString ont;
	bool benchmark;
	bool bm_gpu;
	int bm_iter;
} args_t;



/**
 * Run a benchmark of the FDL algorithm for CPU.
 *
 * @param db
 * @param num_iterations
 */
void benchmark_cpu(Database& db, int num_iterations)
{
	for ( int t = 0; t < num_iterations; t++ )
	{
		for ( Graph *g : db.graphs().values() )
		{
			fdl_2d_cpu(
				g->nodes().size(),
				g->positions().data(),
				g->velocities().data(),
				g->edge_matrix().data()
			);
		}

		printf("%d\n", t + 1);
	}
}



/**
 * Run a benchmark of the FDL algorithm for CPU.
 *
 * @param db
 * @param num_iterations
 */
void benchmark_gpu(Database& db, int num_iterations)
{
	for ( int t = 0; t < num_iterations; t++ )
	{
		for ( Graph *g : db.graphs().values() )
		{
			// execute FDL kernel on GPU
			fdl_2d_gpu(
				g->nodes().size(),
				g->positions_gpu(),
				g->velocities_gpu(),
				g->edge_matrix_gpu()
			);

			// read position data from GPU
			g->gpu_read_positions();
		}

		// wait for GPU to process all graphs
		CUDA_SAFE_CALL(cudaStreamSynchronize(0));

		printf("%d\n", t + 1);
	}
}



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
		"  --benchmark    run the FDL benchmark\n"
		"  --bm-gpu       run the FDL benchmark for GPU\n"
		"  --bm-iter      the number of iterations to run the benchmark\n"
		"  --help         list help options\n"
	);
}



int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	// parse command-line arguments
	args_t args = {
		"config/test_M-R.json",
		"go-basic.obo",
		false,
		false,
		1000
	};

	QStringList options = app.arguments();

	for ( int i = 1; i < options.size(); i++ )
	{
		if ( options[i] == "--config" )
		{
			args.config = options[i + 1];
			i++;
		}
		else if ( options[i] == "--ont" )
		{
			args.ont = options[i + 1];
			i++;
		}
		else if ( options[i] == "--benchmark" )
		{
			args.benchmark = true;
		}
		else if ( options[i] == "--bm-gpu" )
		{
			args.bm_gpu = true;
		}
		else if ( options[i] == "--bm-iter" )
		{
			args.bm_iter = options[i + 1].toInt();
			i++;
		}
		else if ( options[i] == "--help" )
		{
			print_usage();
			exit(0);
		}
		else
		{
			print_usage();
			exit(1);
		}
	}

	// load data
	Database db;
	db.load_config(args.config);
	db.load_ontology(args.ont);

	// run benchmark
	if ( args.benchmark )
	{
		if ( args.bm_gpu )
		{
			benchmark_gpu(db, args.bm_iter);
		}
		else
		{
			benchmark_cpu(db, args.bm_iter);
		}

		return 0;
	}

	// otherwise run the GUI application as normal
	else
	{
		// initialize OpenGL format
		QSurfaceFormat format;
		format.setDepthBufferSize(24);
		format.setVersion(3, 2);
		format.setProfile(QSurfaceFormat::CoreProfile);
		QSurfaceFormat::setDefaultFormat(format);

		// initialize window
		MainWindow window(&db);
		window.showMaximized();

		return app.exec();
	}
}
