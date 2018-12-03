#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include <QString>



class Arguments
{
private:
	static Arguments* _instance;

public:
	static Arguments& instance();
	Arguments() = default;
	Arguments(const Arguments&) = delete;

public:
	QString config;
	QString ont;
	bool fdl;
	bool fdl_3d;
	bool fdl_gpu;
	int fdl_iter;
};



#endif
