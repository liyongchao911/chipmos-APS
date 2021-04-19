#ifndef __CHROMOSOME_BASE_H__
#define __CHROMOSOME_BASE_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <include/job_base.h>
#include <include/machine_base.h>

typedef struct chromosome_base_t chromosome_base_t;
typedef struct chromosome_base_operations_t chromosome_base_operations_t;

struct chromosome_base_t{
	int chromosome_no;
	size_t gene_size;
	double * ms_genes;
	double * os_genes;
	double fitnessValue;
	double *genes;
};

struct chromosome_base_operations_t{
	void (*init)(void *self, double *address);
	void (*reset)(void *self);
	void (*computeFitnessValue)(void *self, machine_base_t *machines, unsigned int machine_sizes, machine_base_operations_t *op);
};


chromosome_base_t * createChromosomeBase(size_t gene_size);

__device__ __host__ void resetChromosomeBase(chromosome_base_t *base);

__device__ __host__ void initChromosomeBase(chromosome_base_t * base, double *address);


#endif
