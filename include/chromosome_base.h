#ifndef __CHROMOSOME_BASE_H__
#define __CHROMOSOME_BASE_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <include/machine_base.h>

class ChromosomeBase{
protected:
	unsigned int chromosome_no;
	unsigned int gene_size;
	double * genes;
	double * ms_genes;
	double * os_genes;
	double fitnessValue;
public:
	ChromosomeBase(unsigned int no);
	__device__ __host__ void init();
	__device__ __host__ void setGene(double *genes);
	__device__ __host__ double const * getMsGenePointer(unsigned int index);
	__device__ __host__ double const * getOsGenePointer(unsigned int index);
	__device__ __host__ double getFitnessValue();
	__device__ __host__ virtual double computeFitnessValue(MachineBase *)=0;
};
#endif
