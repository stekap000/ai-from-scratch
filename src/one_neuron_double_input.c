#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define array_size(arr) sizeof(arr)/sizeof(arr[0])

float train[][2] = {
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
};

float rand_float(void) {
	return (float)rand() / (float)RAND_MAX;
}

float cost(float w, float b) {
	float result = 0.0f;
	for(size_t i = 0; i < array_size(train); ++i) {
		float x = train[i][0];
		float y = w*x + b;
		float d = y - train[i][1];
		result += d*d;
	}

	result /= array_size(train);
	return result;
}

// Assumed model for the problem: y = w*x
int main(void) {
	srand(time(0)); rand();

	float w = rand_float() * 10.f;
	float b = rand_float() * 5.f;
	
	float eps = 1e-3;
	float learning_rate = 1e-3;
	
	printf("Cost: %f\n", cost(w, b));
	for(int i = 0; i < 10000; ++i) {
		float c = cost(w, b);
		float cost_derivative_w = (cost(w + eps, b) - c) / eps;
		float cost_derivative_b = (cost(w, b + eps) - c) / eps;
		w -= cost_derivative_w*learning_rate;
		b -= cost_derivative_b*learning_rate;
	}
	
	printf("Cost: %f\n", cost(w, b));
	printf("w: %f\n", w);
	printf("b: %f\n", b);

	return 0;
}
