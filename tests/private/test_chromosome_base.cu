#include <include/chromosome_base.h>
#include <tests/include/test_chromosome_base.h>

Chromosome * createChromosome(size_t gene_size){
	Chromosome * chromosome = (Chromosome *)malloc(sizeof(Chromosome) + sizeof(double)*gene_size);
	if(!chromosome)
		return NULL;

	initChromosomeBase(&chromosome->base, NULL);
	return chromosome;
}
