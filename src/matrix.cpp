#include <algorithm>
#include <cstring>
#include "matrix.h"



Matrix::Matrix(int rows, int cols):
	_rows(rows),
	_cols(cols),
	_data(new elem_t[(int64_t)rows * cols])
{
}



Matrix::Matrix(Matrix&& M)
	: Matrix()
{
	swap(*this, M);
}



Matrix::~Matrix()
{
	delete[] _data;
}



void Matrix::init_zeros()
{
	Matrix& M = *this;

	memset(M._data, 0, (int64_t)M._rows * M._cols * sizeof(elem_t));
}



void swap(Matrix& A, Matrix& B)
{
	std::swap(A._rows, B._rows);
	std::swap(A._cols, B._cols);
	std::swap(A._data, B._data);
}
