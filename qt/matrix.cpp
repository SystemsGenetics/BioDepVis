#include <algorithm>
#include <cstring>
#include "matrix.h"

Matrix::Matrix(int rows, int cols)
{
	this->_rows = rows;
	this->_cols = cols;
	this->_data = new int[rows * cols];
}

Matrix::Matrix()
{
	this->_rows = 0;
	this->_cols = 0;
	this->_data = nullptr;
}

Matrix::Matrix(Matrix&& M)
	: Matrix()
{
	swap(*this, M);
}

Matrix::~Matrix()
{
	delete[] this->_data;
}

void Matrix::init_zeros()
{
	Matrix& M = *this;

	memset(M._data, 0, M._rows * M._cols * sizeof(int));
}

void swap(Matrix& A, Matrix& B)
{
	std::swap(A._rows, B._rows);
	std::swap(A._cols, B._cols);
	std::swap(A._data, B._data);
}
