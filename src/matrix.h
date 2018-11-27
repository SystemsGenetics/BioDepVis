#ifndef MATRIX_H
#define MATRIX_H

typedef bool matrix_elem_t;



class Matrix
{
private:
	int _rows {0};
	int _cols {0};
	matrix_elem_t *_data {nullptr};

public:
	Matrix(int rows, int cols);
	Matrix(Matrix&& M);
	Matrix() = default;
	~Matrix();

	void init_zeros();

	// getter functions
	int rows() const { return _rows; }
	int cols() const { return _cols; }
	matrix_elem_t * data() const { return _data; }
	matrix_elem_t& elem(int i, int j) const { return _data[i * _cols + j]; }

	// operators
	Matrix& operator=(Matrix B) { swap(*this, B); return *this; }

	// friend functions
	friend void swap(Matrix& A, Matrix& B);
};



#endif
