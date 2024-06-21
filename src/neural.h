#ifndef NEURAL_H
#define NEURAL_H

#include <stdio.h> // For printing.

#ifndef NEURAL_ASSERT
#include <assert.h>
#define NEURAL_ASSERT(cond) assert(cond)
#endif

#ifndef NEURAL_MALLOC
#include <stdlib.h>
#define NEURAL_MALLOC malloc
#endif

#ifndef NEURAL_CALLOC
#include <stdlib.h>
#define NEURAL_CALLOC calloc
#endif

#define NEURAL_NOT_IMPLEMENTED(msg) NEURAL_ASSERT(!(msg))

#ifdef NEURAL_USE_DOUBLE
#define Neural_Real double
#define NEURAL_REAL_PRINT_TYPE "%.4lf"
#else
#define Neural_Real float
#define NEURAL_REAL_PRINT_TYPE "%.4f"
#endif

// Cols first.
typedef struct {
	int rows;
	int cols;
	Neural_Real* elements;
} Matrix;

typedef struct {
	int n;
	Neural_Real* elements;
} Vector;

Matrix matrix_alloc(int rows, int cols);
void matrix_add(Matrix a, Matrix b, Matrix *result);
void matrix_mul(Matrix a, Matrix b, Matrix *result);
void matrix_apply(Matrix a, Vector v, Vector *result);
void matrix_print(Matrix a);

Vector vector_alloc(int n);
void vector_print(Vector v);

#endif // NEURAL_H

#ifdef NEURAL_IMPLEMENTATION

Matrix matrix_alloc(int rows, int cols) {
	return (Matrix) {
		.rows = rows,
		.cols = cols,
		.elements = malloc(rows*cols*sizeof(Neural_Real))
	};
}

void matrix_add(Matrix a, Matrix b, Matrix *result) {
	// Current assumption is that dimensions match for addition.
   	for(int i = 0; i < result->rows * result->cols; ++i) {
		result->elements[i] = a.elements[i] + b.elements[i];
	}
}

void matrix_mul(Matrix a, Matrix b, Matrix *result) {
	NEURAL_NOT_IMPLEMENTED("");
}

void matrix_apply(Matrix a, Vector v, Vector *result) {
	NEURAL_NOT_IMPLEMENTED("");
	
}

void matrix_print(Matrix a) {
	printf("[\n");
	for(int i = 0; i < a.rows; ++i) {
		for(int j = 0; j < a.cols; ++j) {
			printf("\t"NEURAL_REAL_PRINT_TYPE", ", a.elements[i*a.cols + j]);
		}
		printf("\n");
	}
	printf("]\n");
}

Vector vector_alloc(int n) {
	return (Vector) {
		.n = n,
		.elements = malloc(n*sizeof(Neural_Real))
	};	
}

void vector_print(Vector v) {
	printf("[\n");
	for(int i = 0; i < v.n; ++i) {
		printf("\t"NEURAL_REAL_PRINT_TYPE", ", v.elements[i]);
	}
	printf("]\n");	
}

#endif // NEURAL_IMPLEMENTATION
