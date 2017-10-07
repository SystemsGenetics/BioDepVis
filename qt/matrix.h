#ifndef MATRIX_H
#define MATRIX_H

class Matrix {
private:
	int _rows;
	int _cols;
	int *_data;

public:
	Matrix(int rows, int cols);
	Matrix(Matrix&& M);
	Matrix();
	~Matrix();

	void init_zeros();

	// getter functions
	int rows() const { return this->_rows; }
	int cols() const { return this->_cols; }
	int * data() const { return this->_data; }
	int& elem(int i, int j) const { return _data[i * _cols + j]; }

	// operators
	Matrix& operator=(Matrix B) { swap(*this, B); return *this; }

	// friend functions
	friend void swap(Matrix& A, Matrix& B);
};

#endif
