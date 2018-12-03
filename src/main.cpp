#include <QApplication>
#include <QSurfaceFormat>
#include "arguments.h"
#include "database.h"
#include "fdl.h"
#include "mainwindow.h"



/**
 * Run a benchmark of the FDL algorithm for CPU.
 *
 * @param db
 * @param _3d
 * @param num_iterations
 */
void benchmark_cpu(Database& db, bool _3d, int num_iterations)
{
	for ( int t = 0; t < num_iterations; t++ )
	{
		for ( Graph *g : db.graphs().values() )
		{
			if ( _3d )
			{
				fdl_3d_cpu(
					g->nodes().size(),
					g->positions().data(),
					g->velocities().data(),
					g->edge_matrix().data()
				);
			}
			else
			{
				fdl_2d_cpu(
					g->nodes().size(),
					g->positions().data(),
					g->velocities().data(),
					g->edge_matrix().data()
				);
			}
		}

		printf("%d\n", t + 1);
	}
}



/**
 * Run a benchmark of the FDL algorithm for CPU.
 *
 * @param db
 * @param _3d
 * @param num_iterations
 */
void benchmark_gpu(Database& db, bool _3d, int num_iterations)
{
	for ( int t = 0; t < num_iterations; t++ )
	{
		for ( Graph *g : db.graphs().values() )
		{
			// execute FDL kernel on GPU
			if ( _3d )
			{
				fdl_3d_gpu(
					g->nodes().size(),
					g->positions_gpu(),
					g->velocities_gpu(),
					g->edge_matrix_gpu()
				);
			}
			else
			{
				fdl_2d_gpu(
					g->nodes().size(),
					g->positions_gpu(),
					g->velocities_gpu(),
					g->edge_matrix_gpu()
				);
			}

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
		"  --fdl          run the FDL benchmark\n"
		"  --fdl-3d       use 3D FDL\n"
		"  --fdl-gpu      use GPU for FDL benchmark\n"
		"  --fdl-iter     the number of iterations to run the benchmark\n"
		"  --help         list help options\n"
	);
}



int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	// parse command-line arguments
	Arguments& args {Arguments::instance()};
	args.config = "config/test_M-R.json";
	args.ont = "go-basic.obo";
	args.fdl = false;
	args.fdl_3d = false;
	args.fdl_gpu = false;
	args.fdl_iter = 1000;

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
		else if ( options[i] == "--fdl" )
		{
			args.fdl = true;
		}
		else if ( options[i] == "--fdl-3d" )
		{
			args.fdl_3d = true;
		}
		else if ( options[i] == "--fdl-gpu" )
		{
			args.fdl_gpu = true;
		}
		else if ( options[i] == "--fdl-iter" )
		{
			args.fdl_iter = options[i + 1].toInt();
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
	if ( args.fdl )
	{
		if ( args.fdl_gpu )
		{
			benchmark_gpu(db, args.fdl_3d, args.fdl_iter);
		}
		else
		{
			benchmark_cpu(db, args.fdl_3d, args.fdl_iter);
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
