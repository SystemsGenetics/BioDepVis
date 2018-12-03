#include "arguments.h"



Arguments* Arguments::_instance {nullptr};



Arguments& Arguments::instance()
{
	if ( !_instance )
	{
		_instance = new Arguments();
	}
	
	return *_instance;
}
