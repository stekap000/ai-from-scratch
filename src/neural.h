#ifndef NEURAL_H
#define NEURAL_H

#ifndef NEURAL_ASSERT
#include <assert.h>
#define NEURAL_ASSERT(cond) assert(cond)
#endif

#ifndef NEURAL_MALLOC
#include <stdlib.h>
#define NEURAL_MALLOC malloc
#endif

#define NEURAL_NOT_IMPLEMENTED(msg) NEURAL_ASSERT(!(msg))

typedef struct {
	int rows;
	int cols;
	float* elements;
} Matrix;

Matrix matrix_alloc(int rows, int cols);
void matrix_add(Matrix a, Matrix b, Matrix *result);
void matrix_mul(Matrix a, Matrix b, Matrix *result);
void matrix_print(Matrix a);

#endif // NEURAL_H

#ifdef NEURAL_IMPLEMENTATION

Matrix matrix_alloc(int rows, int cols) {
	NEURAL_NOT_IMPLEMENTED("");
	
}

void matrix_add(Matrix a, Matrix b, Matrix *result) {
	NEURAL_NOT_IMPLEMENTED("");
}

void matrix_mul(Matrix a, Matrix b, Matrix *result) {
	NEURAL_NOT_IMPLEMENTED("");
}

void matrix_print(Matrix a) {
	NEURAL_NOT_IMPLEMENTED("");
}

#endif // NEURAL_IMPLEMENTATION
