#define NEURAL_IMPLEMENTATION
#define NEURAL_DEBUG
#include "neural.h"
#include "time.h"
#include "stdbool.h"

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
	//xor.output
	network_print(xor);

	printf("\nCOST BEFORE: %f\n", network_cost(&xor, train));

	for(int i = 0; i < 20000; ++i) {
		Vector g = network_cost_gradient(&xor, train);
		apply_gradient(&xor, g);
		vector_free(&g);
	}

	printf("COST AFTER: %f\n\n", network_cost(&xor, train));
	network_print(xor);
	
	printf("\nValues for training set:\n");
	for(int i = 0; i < train.n; ++i) {
		vector_print(network_forward(&xor, train.samples[i].input));
	}
}

int neural_strcmp(char* s1, char* s2) {
	while(*s1++ == *s2++);
	return *s1 - *s2;
}

// TODO: Add training data allocation to neural.h.
Training_Data parse_iris_data(char* filename) {
	Training_Data data = {
		.n = 150,
		.samples = malloc(150*sizeof(Training_Sample))
	};
	
	FILE* file = fopen(filename, "r");
	if(!file) {
		return data;
	}
	
	char line[128] = {};
	// Ignore column description.
	fgets(line, 128, file);

	bool parsing_string = false;
	int buff_index = 0;
	char buff[16] = {};
	int sample_number = 0;
	int input_number = 0;
	while(fgets(line, 128, file) != 0) {
		input_number = 0;
		data.samples[sample_number].input = vector_alloc(4);
		data.samples[sample_number].output = vector_alloc(3);
		for(int i = 0; line[i] != 0; ++i) {
			switch(line[i]) {
			case ',': {
				buff[buff_index] = 0;
				buff_index = 0;
				data.samples[sample_number].input.elements[input_number++] = neural_atof(buff); 
			} break;
			case '\n': {
				buff_index = 0;
			} break;
			case '"': {
				parsing_string = !parsing_string;
				if(!parsing_string) {
					buff[buff_index] = 0;
					
					if(neural_strcmp(buff, "Setosa")) {
						data.samples[sample_number].output.elements[0] = 1;
						data.samples[sample_number].output.elements[1] = 0;
						data.samples[sample_number].output.elements[2] = 0;
					}
					if(neural_strcmp(buff, "Versicolor")) {
						data.samples[sample_number].output.elements[0] = 0;
						data.samples[sample_number].output.elements[1] = 1;
						data.samples[sample_number].output.elements[2] = 0;
					}
					if(neural_strcmp(buff, "Virginica")) {
						data.samples[sample_number].output.elements[0] = 0;
						data.samples[sample_number].output.elements[1] = 0;
						data.samples[sample_number].output.elements[2] = 1;
					}
				}
			} break;
			default: {
				buff[buff_index++] = line[i];
			}
			}
		}

		++sample_number;
	}
		  
	return data;
}

void example_iris() {
	Training_Data data = parse_iris_data("iris.csv");
	//training_data_print(data);

	Network iris = random_network(3, (int[]){4, 5, 3, 3});
	iris.learning_rate = 1e-1;
	iris.eps = 1e-1;
	iris.output_activation_vector_function_mut = activation_vector_function_softmax_mut;
	network_print(iris);

	printf("\nCOST BEFORE: %f\n", network_cost(&iris, data));

	for(int i = 0; i < 100; ++i) {
		Vector g = network_cost_gradient(&iris, data);
		apply_gradient(&iris, g);
		vector_free(&g);
	}
	
	printf("COST AFTER: %f\n\n", network_cost(&iris, data));
	network_print(iris);

	printf("\nValues for training set:\n");
	for(int i = 0; i < data.n; ++i) {
		vector_print(network_forward(&iris, data.samples[i].input));
	}	
}

int main(void) {
	srand(time(0));

	//example_xor();
	example_iris();

	return 0;
}
