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

float cost(float w) {
	float result = 0.0f;
	for(size_t i = 0; i < array_size(train); ++i) {
		float x = train[i][0];
		float y = w*x;
		float d = y - train[i][1];
		result += d*d;
	}

	result /= array_size(train);
	return result;
}

// Assumed model for the problem: y = w*x
int main(void) {
	srand(time(0)); rand();

	float w = rand_float() * 10;
	float eps = 1e-3;
	float learning_rate = 1e-3;
	
	printf("Cost: %f\n", cost(w));
	for(int i = 0; i < 10000; ++i) {
		float cost_derivative = (cost(w + eps) - cost(w)) / eps;
		w -= cost_derivative*learning_rate;
	}
	
	printf("Cost: %f\n", cost(w));
	printf("%f\n", w);

	return 0;
}
