#define NEURAL_IMPLEMENTATION
#define NEURAL_DEBUG
#include "neural.h"
#include "time.h"

void example_xor() {
	Training_Data train = {
		.n = 4,
		.samples = (Training_Sample[]) {
			{
				{ 2, (Neural_Real[]){0, 0}},
				{ 1, (Neural_Real[]){0}}
			},
			{
				{ 2, (Neural_Real[]){0, 1}},
				{ 1, (Neural_Real[]){1}}
			},
			{
				{ 2, (Neural_Real[]){1, 0}},
				{ 1, (Neural_Real[]){1}}
			},
			{
				{ 2, (Neural_Real[]){1, 1}},
				{ 1, (Neural_Real[]){0}}
			},
		}
	};

	Network xor = random_network(2, (int[]){2, 2, 1});
	xor.learning_rate = 1e-1;
	xor.eps = 1e-1;
	network_print(xor);

	for(int i = 0; i < 20000; ++i) {
		Vector g = network_cost_gradient(&xor, train);
		apply_gradient(&xor, g);
		vector_free(&g);
	}

	printf("\nCOST AFTER: %f\n\n", network_cost(&xor, train));
	network_print(xor);
	
	printf("\nValues for training set:\n");
	for(int i = 0; i < train.n; ++i) {
		vector_print(network_forward(&xor, train.samples[i].input));
	}
}

int main(void) {
	srand(time(0));

	example_xor();
	
	return 0;
}
