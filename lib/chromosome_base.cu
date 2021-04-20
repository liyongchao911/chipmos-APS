#include <include/chromosome_base.h>
#include <stdlib.h>
#include "../include/chromosome_base.h"

chromosome_base_t * createChromosomeBase(size_t gene_size, int chromosome_no){
	chromosome_base_t * chromosome = (chromosome_base_t*)malloc(sizeof(chromosome_base_t) + sizeof(double) * gene_size);
	
	if(!chromosome)
		return NULL;
	
	initChromosomeBase(chromosome, NULL);
	chromosome->gene_size = gene_size;	
	chromosome->chromosome_no = chromosome_no;
	return chromosome;
}
__device__ __host__ void resetChromosomeBase(chromosome_base_t *base){
	base->fitnessValue = 0;
}

__device__ __host__ void initChromosomeBase(chromosome_base_t *base, double* address){
	int mid = base->gene_size >> 1;
	if(address){
		base->genes = address;
	}
	base->ms_genes = base->genes;
	base->os_genes = base->ms_genes + mid;
	resetChromosomeBase(base);
}
