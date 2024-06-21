#define NEURAL_IMPLEMENTATION
#include "neural.h"

int main() {
	Matrix a = matrix_alloc(3, 4);

	a.elements[5] = 1.23;
	a.elements[2] = 212.23;
	a.elements[11] = 1.0922;
	a.elements[8] = 11.23;

	matrix_print(a);
	
	return 0;
}
