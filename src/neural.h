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

typedef struct {
	int n;
	Neural_Real* elements;
} Vector;

// Rows first, for now.
typedef struct {
	int rows;
	int cols;
	Neural_Real* elements;
} Matrix;

/*
Layer model:
[previous_layer_output_vector] *
|w0 w1 w2 ... size(previous_layer_output_vector)|
|...                                            |
|...                                            |
|size(current_layer_vector)                     | +
[biases (one for each neuron in current layer)]

Short description:
current_layer_output_vector = activation(input_later_output_vector*weights + biases)
activation(W*A + B)
*/
typedef struct {
	int layers_num;
	// Has (layers_num + 1) elements (for input and layers).
	// First element is the size of input.
	int* layers_sizes;
	// Has (layers_num) elements.
	Matrix* weight_matrices;
	// Has (layers_num) elements.
	Vector* bias_vectors;
	int max_layer_size;
} Network;

Vector vector_alloc(int n);
void vector_free(Vector* v);
void vector_add(Vector u, Vector v, Vector* result);
void vector_elements(Vector* v, Neural_Real elements[]);
void vector_print(Vector v);

Matrix matrix_alloc(int rows, int cols);
void matrix_free(Matrix* m);
void matrix_add(Matrix a, Matrix b, Matrix *result);
void matrix_mul(Matrix a, Matrix b, Matrix *result);
void matrix_vector_mul(Matrix a, Vector v, Vector *result);
void matrix_print(Matrix a);

Network network_alloc(int layers_num, int* layers_sizes);
void network_layer(Network* n, int layer_index, Neural_Real weights_elements[], Neural_Real biases_elements[]);
Vector network_create_input_vector(Network n, Neural_Real elements[]);
Vector network_forward(Network n, Vector v);
void network_print(Network n);

#endif // NEURAL_H

#ifdef NEURAL_IMPLEMENTATION

Vector vector_alloc(int n) {
	return (Vector) {
		.n = n,
		.elements = NEURAL_CALLOC(n, sizeof(Neural_Real))
	};	
}

void vector_free(Vector* v) {
	free(v->elements);
	v->elements = 0;
	v->n = 0;
}

void vector_add(Vector u, Vector v, Vector* result) {
	for(int i = 0; i < u.n; ++i)
		result->elements[i] = u.elements[i] + v.elements[i];
}

void vector_elements(Vector* v, Neural_Real elements[]) {
	for(int i = 0; i < v->n; ++i)
		v->elements[i] = elements[i];
}

void vector_print(Vector v) {
	printf("[");
	for(int i = 0; i < v.n; ++i) {
		printf("\t"NEURAL_REAL_PRINT_TYPE", ", v.elements[i]);
	}
	printf("]\n");
}

Matrix matrix_alloc(int rows, int cols) {
	return (Matrix) {
		.rows = rows,
		.cols = cols,
		.elements = NEURAL_CALLOC(rows*cols, sizeof(Neural_Real))
	};
}

void matrix_free(Matrix* m) {
	free(m->elements);
	m->elements = 0;
	m->rows = 0;
	m->cols = 0;
}

void matrix_add(Matrix a, Matrix b, Matrix *result) {
	// Current assumption is that dimensions match for addition.
   	for(int i = 0; i < result->rows * result->cols; ++i) {
		result->elements[i] = a.elements[i] + b.elements[i];
	}
}

void matrix_mul(Matrix a, Matrix b, Matrix *result) {
	NEURAL_NOT_IMPLEMENTED("");
	(void)a;
	(void)b;
	(void)result;
}

void matrix_vector_mul(Matrix a, Vector v, Vector *result) {
	// Current assumption is that dimensions match for matrix vector multiplication.
	float temp = 0;
	for(int i = 0; i < a.rows; ++i) {
		temp = 0;
		for(int j = 0; j < a.cols; ++j) {
			temp += a.elements[i*a.cols + j] * v.elements[j];
		}
		result->elements[i] = temp;
	}
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

Network network_alloc(int layers_num, int layers_sizes[]) {
	Network n;
	n.layers_num = layers_num;
	n.layers_sizes = NEURAL_CALLOC(layers_num + 1, sizeof(n.layers_sizes[0]));
	n.weight_matrices = NEURAL_CALLOC(layers_num, sizeof(Matrix));
	n.bias_vectors = NEURAL_CALLOC(layers_num, sizeof(Vector));
	n.max_layer_size = 0;

	for(int i = 0; i < layers_num; ++i) {
		n.layers_sizes[i] = layers_sizes[i];
		n.weight_matrices[i] = matrix_alloc(layers_sizes[i+1], layers_sizes[i]);
		n.bias_vectors[i] = vector_alloc(layers_sizes[i+1]);

		if(layers_sizes[i] > n.max_layer_size)
			n.max_layer_size = layers_sizes[i];
	}
		
	if(layers_sizes[layers_num] > n.max_layer_size)
		n.max_layer_size = layers_sizes[layers_num + 1];

	return n;
}

void network_layer(Network* n, int layer_index, Neural_Real weights_elements[], Neural_Real biases_elements[]) {
#define M n->weight_matrices[layer_index]
	for(int i = 0; i < M.rows * M.cols; ++i) {
		M.elements[i] = weights_elements[i];
	}
#undef M
#define B n->bias_vectors[layer_index]
	for(int i = 0; i < B.n; ++i) {
		B.elements[i] = biases_elements[i];
	}
#undef B
}

Vector network_create_input_vector(Network n, Neural_Real elements[]) {
	Vector v = vector_alloc(n.max_layer_size);
	vector_elements(&v, elements);
	return v;
}

Vector network_forward(Network n, Vector v) {
	Vector temp = vector_alloc(v.n);
	for(int i = 0; i < n.layers_num; ++i) {
		matrix_vector_mul(n.weight_matrices[i], v, &temp);
		vector_elements(&v, temp.elements);
		v.n = n.weight_matrices[i].rows;
		vector_add(v, n.bias_vectors[i], &v);
	}
	vector_free(&temp);
	return v;
}

void network_print(Network n) {
	printf("Network:\n");
	for(int i = 0; i < n.layers_num; ++i) {
		printf("Layer %d: ==============================\n", i);
		printf("Weights:\n");
		matrix_print(n.weight_matrices[i]);
		printf("Biases:\n");
		vector_print(n.bias_vectors[i]);
	}
}

#endif // NEURAL_IMPLEMENTATION
