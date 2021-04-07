#include <include/common.h>

#define greater(a, b, type) *(type*)a > *(type*)b

int cmpint(const void *a, const void *b){
	return greater(a, b, int);
}

double cmpdouble(const void *a, const void *b){
	return greater(a, b, double);
}
