#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Since XOR can be represented as (x|y) & ~(x&y), we can use it to "guess" the structure
// of model. We will have two inputs x and y. Middle layer will consist of "OR" and "NAND"
// nodes, which will then be fed to "AND" node. These are just nodes, and they do not
// perform any kind of logic function at the beginning. We are just calling them like that
// to explicitly keep track of them.

// Idea is to have train data for XOR logic function, plus this "guessed" structure of
// of model. As a result, we expect to see nodes behaving as a combination of logic
// function which give rise to XOR.

// One thing to note is that in the case of a more complicated problem, with unknown
// solution, we wouldn't have that clear indication for the structure of network, but
// we would have various hints based on what we want to model.

// When created, this structure is not just valid for XOR, but also for AND, OR, etc. ie.
// we can use it to train network to represent those logic functions (and many others).

// In other words, our program defines a general machine that we can load (program) with
// data. By injecting data, we are constructing specific machine.

#define model_param_num(model) (sizeof(model)/sizeof(float))
#define model_param_ptr(model, idx) (((float*)&(model)) + idx)
#define model_param_val(model, idx) (*(((float*)&(model)) + idx))
#define foreach_param_in_model(model) for(float* it = (float*)&(model); (size_t)it != (size_t)((model_param_ptr(model, model_param_num(model)))); ++it)

typedef struct {
	float or_w1;
	float or_w2;
	float or_b;
	float nand_w1;
	float nand_w2;
	float nand_b;
	float and_w1;
	float and_w2;
	float and_b;
} Xor;

float sigmoidf(float x) {
	return 1.f / (1.f + expf(-x));
}

float forward(Xor m, float x1, float x2) {
	float a = sigmoidf(x1*m.or_w1 + x2*m.or_w2 + m.or_b);
	float b = sigmoidf(x1*m.nand_w1 + x2*m.nand_w2 + m.nand_b);
	return sigmoidf(a*m.and_w1 + b*m.and_w2 + m.and_b);
}

typedef float sample[3];
float xor_train[][3] = {
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
};

sample *train = xor_train;
size_t train_count = 4;

float cost(Xor m) {
	float result = 0.0f;
	for(size_t i = 0; i < train_count; ++i) {
		float x1 = train[i][0];
		float x2 = train[i][1];
		float y = forward(m, x1, x2);
		float d = y - train[i][2];
		result += d*d;
	}

	result /= train_count;
	return result;
}

float rand_float(void) {
	return (float)rand() / (float)RAND_MAX;
}

Xor rand_xor(void) {
	Xor m;
	foreach_param_in_model(m) {
		*it = rand_float();
	}
	return m;
}

void print_xor(Xor m) {
	printf("or_w1   = %f\n", m.or_w1);
	printf("or_w2   = %f\n", m.or_w2);
	printf("or_b    = %f\n", m.or_b);
	printf("nand_w1 = %f\n", m.nand_w1);
	printf("nand_w2 = %f\n", m.nand_w2);
	printf("nand_b  = %f\n", m.nand_b);
	printf("and_w1  = %f\n", m.and_w1);
	printf("and_w2  = %f\n", m.and_w2);
	printf("and_b   = %f\n", m.and_b);
}

void print_xor_logic_functions(Xor m) {
	printf("Model:\n");
	for(size_t i = 0; i < train_count; ++i) {
		int x1 = train[i][0];
		int x2 = train[i][1];
		float v = forward(m, x1, x2);
		printf("\t%d XOR %d = %f\n", x1, x2, v);
	}
	printf("Neuron 1:\n");
	for(size_t i = 0; i < train_count; ++i) {
		float v = sigmoidf(m.or_w1 * train[i][0] + m.or_w2 * train[i][1] + m.or_b);
		printf("\t%d | %d = %f\n", (int)train[i][0], (int)train[i][1], v);
	}
	printf("Neuron 2:\n");
	for(size_t i = 0; i < train_count; ++i) {
		float v = sigmoidf(m.nand_w1 * train[i][0] + m.nand_w2 * train[i][1] + m.nand_b);
		printf("\t%d | %d = %f\n", (int)train[i][0], (int)train[i][1], v);
	}
	printf("Neuron 3:\n");
	for(size_t i = 0; i < train_count; ++i) {
		float v = sigmoidf(m.and_w1 * train[i][0] + m.and_w2 * train[i][1] + m.and_b);
		printf("\t%d | %d = %f\n", (int)train[i][0], (int)train[i][1], v);
	}
}

Xor model_gradient(Xor m, float eps) {
	Xor g;
	float c = cost(m);
	float saved;

	for(size_t i = 0; i < sizeof(m)/sizeof(float); ++i) {
		saved = *((float*)&m + i);
		*model_param_ptr(m, i) += eps;
		*model_param_ptr(g, i) = (cost(m) - c) / eps;
		*model_param_ptr(m, i) = saved;
	}
	
	return g;
}

Xor apply_gradient(Xor m, Xor g, float learning_rate) {
	for(size_t i = 0; i < sizeof(m)/sizeof(float); ++i) {
		*model_param_ptr(m, i) -= model_param_val(g, i)*learning_rate;
	}
	return m;
}

int main(void) {
	srand(time(0));

	float eps = 1e-1;
	float learning_rate = 1e-1;
	
	Xor m = rand_xor();
	Xor g;
	
	print_xor(m);
	printf("cost = %f\n\n", cost(m));
	
	for(int n = 0; n < 100000; ++n) {
		g = model_gradient(m, eps);
		m = apply_gradient(m, g, learning_rate);
	}
	
	print_xor(m);
	printf("cost = %f\n\n", cost(m));

	print_xor_logic_functions(m);
	
	return 0;
}
