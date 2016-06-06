#include<unordered_map>
#include<iostream>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<vector>
#include<string>
#include <time.h>
#include "G3NA.h"



 void readOntFile(std::unordered_map<std::string, ontStruct> *array)
{
	
	std::string line, word;
	ontStruct *a = new ontStruct();
	int count = 0;
	time_t t1;
	time_t t2;
	double sec;

	t1 = time(NULL);
	std::ifstream my1("ont_data.txt");
	if (my1.is_open())
	{
		while (std::getline(my1, line))
		{
			std::stringstream linestream(line);
			std::getline(linestream, word, ':');

			//Checking for id
			if (word == "id")
			{
				linestream >> a->id;
			}

			//Checking for name
			if (word == "name")
			{
				std::string word1, word2;
				linestream >> word1;
				linestream >> word2;
				a->name = word1;
				while (word1 != word2)
				{
					a->name.append(" ");
					a->name.append(word2);
					word1 = word2;
					linestream >> word2;
				}
			}

			//Checking for def
			if (word == "def")
			{
				std::string word1, word2;
				linestream >> word1;
				linestream >> word2;
				a->def = word1;
				while (word1 != word2)
				{
					a->def.append(" ");
					a->def.append(word2);
					word1 = word2;
					linestream >> word2;

					//Lines below are code to enable to locate the square bracket words in def in case they need to be removed.
					//if(word2[0] == '[')
					//std::cout << "Can locate" << std::endl;
				}
				a->index = count;
				std::pair<std::string, ontStruct> item(a->id, *a);
				array->insert(item);
				count++;
			}
		}
	}
	my1.close();

	t2 = time(NULL);
	sec = difftime(t2, t1);
	std::cout << sec << std::endl;

	//Test to print values for file ont_data1.txt
	//Do not print values for ont_data.txt
	/*for(int i = 0; i < 3; i++)
	{
	std::cout << "ID: " << array[i].id << std::endl;
	std::cout << "Name: " << array[i].name << std::endl;
	std::cout << "Def: " << array[i].def << std::endl;
	}*/

}
