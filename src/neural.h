#ifndef NEURAL_H
#define NEURAL_H

// TODO: Backpropagation.
// TODO: Network to file serialization.
// TODO: Network loading from file.

#include <stdio.h> // For printing.
#include <math.h>
#include <stdlib.h>

#include <stdint.h>
#define u8 uint8_t
#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t

#ifndef NEURAL_ASSERT
#include <assert.h>
#define NEURAL_ASSERT(cond) assert(cond)
#endif

#ifndef NEURAL_MALLOC
#define NEURAL_MALLOC malloc
#endif

#ifndef NEURAL_CALLOC
#define NEURAL_CALLOC calloc
#endif

#define NEURAL_NOT_IMPLEMENTED(msg) NEURAL_ASSERT(!(msg))

#ifdef NEURAL_USE_DOUBLE
#define Neural_Real double
#define NEURAL_REAL_MAX 1.7976931348623158e+308
#define NEURAL_REAL_PRINT_TYPE "%.4lf"
#define NEURAL_REAL_SIGN_BIT_MASK 0x7fffffffffffffff
#define NEURAL_REAL_BITS_TYPE u64
#define neural_exp(x) exp(x)
#else
#define Neural_Real float
#define NEURAL_REAL_MAX 3.402823466e+38F
#define NEURAL_REAL_PRINT_TYPE "%.4f"
#define NEURAL_REAL_SIGN_BIT_MASK 0x7fffffff
#define NEURAL_REAL_BITS_TYPE u32
#define neural_exp(x) expf(x)
#endif

#define neural_atof(str) (Neural_Real)atof(str)

#define NEURAL_REAL_MIN -NEURAL_REAL_MAX

#define NEURAL_DEFAULT_LEARNING_RATE 1e-2
#define NEURAL_DEFAULT_EPS 1e-2

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

typedef Neural_Real (*Activation_Function) (Neural_Real);
typedef void (*Activation_Vector_Function_Mut) (Vector);

typedef struct {
	Vector input;
	Vector output;
} Training_Sample;

// TODO: When vectors are stored as training data, there is no need to have sizes for every
// single vector since they are all of the same size.
typedef struct {
	Training_Sample* samples;
	int n;
} Training_Data;

typedef struct {
	int layers_num;
	int max_layer_size;
	// Has (layers_num + 1) elements (for input and layers).
	// First element is the size of input.
	int* layers_sizes;
	// Has (layers_num) elements.
	Matrix* weight_matrices;
	// Has (layers_num) elements.
	Vector* bias_vectors;
	Activation_Vector_Function_Mut hidden_activation_vector_function_mut;
	Activation_Vector_Function_Mut output_activation_vector_function_mut;
	Neural_Real learning_rate;
	Neural_Real eps;
	int number_of_parameters;
} Network;

Vector vector_alloc(int n);
void vector_free(Vector* v);
void vector_add(Vector u, Vector v, Vector* result);
void vector_elements(Vector* v, Neural_Real elements[]);
void vector_apply_activation_function(Vector* v, Activation_Function activation_function);
void vector_print(Vector v);

Matrix matrix_alloc(int rows, int cols);
void matrix_free(Matrix* m);
void matrix_add(Matrix a, Matrix b, Matrix *result);
void matrix_mul(Matrix a, Matrix b, Matrix *result);
void matrix_vector_mul(Matrix a, Vector v, Vector *result);
void matrix_print(Matrix a);

Network network_alloc(int layers_num, int layers_sizes[]);
void network_free(Network* n);
void network_layer(Network* n, int layer_index, Neural_Real weights_elements[], Neural_Real biases_elements[]);
Vector network_create_input_vector(Network n, Neural_Real elements[]);
Vector network_forward(Network* n, Vector v);
Neural_Real network_cost(Network* n, Training_Data d);
void network_save(Network* n, char* filename);
Network network_load(char* filename);
void network_print(Network n);

float random_neural_real();
Vector random_vector(int n);
Matrix random_matrix(int rows, int cols);
Network random_network(int layers_num, int layers_sizes[]);

Neural_Real activation_function_sigmoid(Neural_Real x);
Neural_Real activation_function_identity(Neural_Real x);
Neural_Real activation_function_relu(Neural_Real x);

void activation_vector_function_sigmoid_mut(Vector v);
void activation_vector_function_identity_mut(Vector v);
void activation_vector_function_relu_mut(Vector v);
Vector activation_vector_function_softmax(Vector v);
void activation_vector_function_softmax_mut(Vector v);

void training_data_print(Training_Data d);

Neural_Real neural_abs(Neural_Real x);

// TODO: This should be done with backpropagation, but for testing purposes of other code,
// it is currently implemented in this slow way.
#ifdef NEURAL_DEBUG
Vector network_cost_gradient(Network* n, Training_Data d) {
	Neural_Real temp = 0;
	Neural_Real base_cost = network_cost(n, d);
	Vector gradient = vector_alloc(n->number_of_parameters);
	int gradient_index = 0;
	for(int i = 0; i < n->layers_num; ++i) {
		Matrix m = n->weight_matrices[i];
		for(int j = 0; j < m.rows*m.cols; ++j) {
			temp = m.elements[j];
			m.elements[j] += n->eps;
			gradient.elements[gradient_index++] = (network_cost(n, d) - base_cost) / n->eps;
			m.elements[j] = temp;
		}

		Vector v = n->bias_vectors[i];
		for(int j = 0; j < v.n; ++j) {
			temp = v.elements[j];
			v.elements[j] += n->eps;
			gradient.elements[gradient_index++] = (network_cost(n, d) - base_cost) / n->eps;
			v.elements[j] = temp;
		}
	}
	
	return gradient;
}

void apply_gradient(Network* n, Vector gradient) {
	int gradient_index = 0;
	for(int i = 0; i < n->layers_num; ++i) {
		Matrix m = n->weight_matrices[i];
		for(int j = 0; j < m.rows*m.cols; ++j)
			m.elements[j] -= gradient.elements[gradient_index++] * n->learning_rate;

		Vector v = n->bias_vectors[i];
		for(int j = 0; j < v.n; ++j)
			v.elements[j] -= gradient.elements[gradient_index++] * n->learning_rate;
	}
}
#endif

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

void vector_apply_activation_function(Vector* v, Activation_Function activation_function) {
	for(int i = 0; i < v->n; ++i) {
		v->elements[i] = activation_function(v->elements[i]);
	}
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
	n.max_layer_size = 0;
	n.layers_sizes = NEURAL_CALLOC(layers_num + 1, sizeof(n.layers_sizes[0]));
	n.weight_matrices = NEURAL_CALLOC(layers_num, sizeof(Matrix));
	n.bias_vectors = NEURAL_CALLOC(layers_num, sizeof(Vector));
	n.hidden_activation_vector_function_mut = activation_vector_function_sigmoid_mut;
	n.output_activation_vector_function_mut = activation_vector_function_sigmoid_mut;
	n.learning_rate = NEURAL_DEFAULT_LEARNING_RATE;
	n.eps = NEURAL_DEFAULT_EPS;
	n.number_of_parameters = 0;
	for(int i = 0; i < layers_num; ++i) {
		n.number_of_parameters += layers_sizes[i]*layers_sizes[i+1] + layers_sizes[i+1]; 
	}
	
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

void network_free(Network* n) {
	for(int i = 0; i < n->layers_num; ++i) {
		matrix_free(n->weight_matrices + i);
		vector_free(n->bias_vectors + i);
	}
	free(n->weight_matrices);
	n->weight_matrices = 0;
	free(n->bias_vectors);
	n->bias_vectors = 0;
	free(n->layers_sizes);
	n->layers_sizes = 0;
	n->layers_num = 0;
	n->max_layer_size = 0;
	n->hidden_activation_vector_function_mut = activation_vector_function_sigmoid_mut;
	n->output_activation_vector_function_mut = activation_vector_function_sigmoid_mut;
	n->learning_rate = NEURAL_DEFAULT_LEARNING_RATE;
	n->eps = NEURAL_DEFAULT_EPS;
	n->number_of_parameters = 0;
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

// This function returns vector that must be freed.
Vector network_forward(Network* n, Vector v_orig) {
	// TODO: This must be less dumb.
	Vector v = vector_alloc(v_orig.n);
	vector_elements(&v, v_orig.elements);
	Vector temp = vector_alloc(v.n);
	for(int i = 0; i < n->layers_num - 1; ++i) {
		matrix_vector_mul(n->weight_matrices[i], v, &temp);
		vector_elements(&v, temp.elements);
		v.n = n->weight_matrices[i].rows;
		vector_add(v, n->bias_vectors[i], &v);
		n->hidden_activation_vector_function_mut(v);
	}

	matrix_vector_mul(n->weight_matrices[n->layers_num - 1], v, &temp);
	vector_elements(&v, temp.elements);
	v.n = n->weight_matrices[n->layers_num - 1].rows;
	vector_add(v, n->bias_vectors[n->layers_num - 1], &v);
	n->output_activation_vector_function_mut(v);
	
	vector_free(&temp);
	return v;
}

Neural_Real network_cost(Network* n, Training_Data d) {
	Neural_Real sum = 0;
	for(int i = 0; i < d.n; ++i) {
		Vector actual_output = network_forward(n, d.samples[i].input);
		for(int j = 0; j < actual_output.n; ++j) {
			Neural_Real dist = (actual_output.elements[j] - d.samples[i].output.elements[j]);
			sum += dist*dist;
		}
		vector_free(&actual_output);
	}
	return sum / d.n;
}

void network_save(Network* n, char* filename) {
	NEURAL_NOT_IMPLEMENTED("");
	(void)n;
	(void)filename;
}

Network network_load(char* filename) {
	NEURAL_NOT_IMPLEMENTED("");
	(void)filename;
}

void network_print(Network n) {
	printf("NETWORK:\n");
	printf("Number of parameters: %d\n", n.number_of_parameters);
	printf("Learning rate: "NEURAL_REAL_PRINT_TYPE"\n", n.learning_rate);
	printf("Eps: "NEURAL_REAL_PRINT_TYPE"\n", n.eps);
	for(int i = 0; i < n.layers_num; ++i) {
		printf("Layer %d: ==============================\n", i);
		printf("Weights:\n");
		matrix_print(n.weight_matrices[i]);
		printf("Biases:\n");
		vector_print(n.bias_vectors[i]);
	}
}

float random_neural_real() {
	return (Neural_Real)rand()/(Neural_Real)RAND_MAX;
}

Vector random_vector(int n) {
	Vector v = vector_alloc(n);
	for(int i = 0; i < n; ++i)
		v.elements[i] = random_neural_real();
	return v;
}

Matrix random_matrix(int rows, int cols) {
	Matrix m = matrix_alloc(rows, cols);
	for(int i = 0; i < m.rows*m.cols; ++i)
		m.elements[i] = random_neural_real();
	return m;
}

Network random_network(int layers_num, int layers_sizes[]) {
	Network n = network_alloc(layers_num, layers_sizes);
	for(int i = 0; i < layers_num; ++i) {
		for(int j = 0; j < n.weight_matrices[i].rows*n.weight_matrices[i].cols; ++j)
			n.weight_matrices[i].elements[j] = random_neural_real();

		for(int j = 0; j < n.bias_vectors[i].n; ++j)
			n.bias_vectors[i].elements[j] = random_neural_real();
	}
	return n;
}

Neural_Real activation_function_sigmoid(Neural_Real x) {
	return 1.f / (1.f + neural_exp(-x));
}

Neural_Real activation_function_identity(Neural_Real x) {
	return x;
}

Neural_Real activation_function_relu(Neural_Real x) {
	return (x + neural_abs(x)) * 0.5;
}

void activation_vector_function_sigmoid_mut(Vector v) {
	for(int i = 0; i < v.n; ++i)
		v.elements[i] = (1.f / (1.f + neural_exp(-v.elements[i])));
}

void activation_vector_function_identity_mut(Vector v) {
	(void)v;
}

void activation_vector_function_relu_mut(Vector v) {
	for(int i = 0; i < v.n; ++i)
		v.elements[i] = (v.elements[i] + neural_abs(v.elements[i])) * 0.5;
}

Vector activation_vector_function_softmax(Vector v) {
	Vector result = vector_alloc(v.n);
	Neural_Real softmax_sum = 0;
	for(int i = 0; i < v.n; ++i) {
		result.elements[i] = neural_exp(v.elements[i]);
		softmax_sum += result.elements[i];
	}
	for(int i = 0; i < v.n; ++i) result.elements[i] /= softmax_sum;
	return result;
}

void activation_vector_function_softmax_mut(Vector v) {
	Neural_Real softmax_sum = 0;
	for(int i = 0; i < v.n; ++i) {
		v.elements[i] = neural_exp(v.elements[i]);
		softmax_sum += v.elements[i];
	}
	for(int i = 0; i < v.n; ++i) v.elements[i] /= softmax_sum;
}

void training_data_print(Training_Data d) {
	for(int i = 0; i < d.n; ++i) {
		vector_print(d.samples[i].input);
		vector_print(d.samples[i].output);
	}
}

Neural_Real neural_abs(Neural_Real x) {
	union { Neural_Real f; NEURAL_REAL_BITS_TYPE u; } fbits = {x};
	fbits.u &= NEURAL_REAL_SIGN_BIT_MASK;
	return fbits.f;
}

#endif // NEURAL_IMPLEMENTATION
