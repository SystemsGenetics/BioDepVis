#ifndef MATRIX_H
#define MATRIX_H

#define ELEM(M, i, j) (M)._data[(j) * (M)._rows + (i)]

class Matrix {
private:
	int _rows;
	int _cols;
	float *_data;

public:
	Matrix(int rows, int cols);
	Matrix(Matrix&& M);
	Matrix();
	~Matrix();

	void init_identity();
	void init_zeros();

	// getter functions
	int rows() const { return this->_rows; }
	int cols() const { return this->_cols; }
	float * data() const { return this->_data; }
	float& elem(int i, int j) const { return ELEM(*this, i, j); }

	// operators
	Matrix& operator=(Matrix B) { swap(*this, B); return *this; }

	// friend functions
	friend void swap(Matrix& A, Matrix& B);
};

#endif
