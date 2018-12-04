#include <algorithm>
#include <cstring>
#include "matrix.h"



/**
 * Construct a matrix.
 *
 * @param rows
 * @param cols
 */
Matrix::Matrix(int rows, int cols):
	_rows(rows),
	_cols(cols),
	_data(new elem_t[(int64_t)rows * cols])
{
}



/**
 * Move-construct a matrix.
 *
 * @param M
 */
Matrix::Matrix(Matrix&& M)
	: Matrix()
{
	swap(*this, M);
}



/**
 * Destruct a matrix.
 */
Matrix::~Matrix()
{
	delete[] _data;
}



/**
 * Initialize a matrix to all zeros.
 */
void Matrix::init_zeros()
{
	Matrix& M = *this;

	memset(M._data, 0, (int64_t)M._rows * M._cols * sizeof(elem_t));
}



/**
 * Swap two matrices.
 *
 * @param A
 * @param B
 */
void swap(Matrix& A, Matrix& B)
{
	std::swap(A._rows, B._rows);
	std::swap(A._cols, B._cols);
	std::swap(A._data, B._data);
}
