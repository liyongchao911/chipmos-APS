#ifndef __JOB_BASE_H__
#define __JOB_BASE_H__
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <include/linked_list.h>

/**
 * @brief Store process time and its corresponding machine number.
 *
 */
typedef struct ProcessTime ProcessTime;

struct ProcessTime{
	unsigned int machine_no;
	double process_time;
};

typedef struct JobBase JobBase;
JobBase * newJobBase();

struct JobBase{
	void * ptr_derived_object;

	// genes point to chromosome's gene
	// use double const * type to prevent set the wrong value on gene
	double const * ms_gene;
	double const * os_seq_gene;
	
	// partition is the partition value of roulette.
	// for example : size of can run tools is 10, partition is 1/10
	double partition; 
	
	// process time
	// process_time is an 1-D array
	ProcessTime ** process_time; 
	unsigned int size_of_process_time;
	
	// job information
	unsigned int job_no;
	unsigned int machine_no;
	double arriv_t;
	double start_time;
	double end_time;

	// constructor and initialization
	void (*init)(void *self);
	void (*reset)(void *self);
	
	// setter
	void (*setMsGenePointer)(void *self, double * ms_gene);
	void (*setOsSeqGenePointer)(void *self, double *os_seq_gene);
	void (*setProcessTime)(void *self, ProcessTime **, unsigned int size_of_process_time);
	void (*setArrivT)(void *self, double arrivT);
	void (*setStartTime)(void *self, double startTime);

	// getter
	double (*getMsGene)(void *self);
	double (*getOsSeqGene)(void *self);
	double (*getArrivT)(void *self);
	double (*getStartTime)(void *self);
	double (*getEndTime)(void *self);
	unsigned int (*getMachineNo)(void *self);

	// operation
	unsigned int (*machineSelection)(void *self);
};

__device__ __host__ void initJobBase(void *self);
__device__ __host__ void resetJobBase(void *self);
__device__ __host__ void setMsGenePointer(void *self, double *ms_gene);
__device__ __host__ void setOsSeqGenePointer(void *self, double *os_seq_gene);
__device__ __host__ void setProcessTime(void *self, ProcessTime ** pt, unsigned int size_of_process_time);
__device__ __host__ void setArrivT(void *self, double arrivT);
__device__ __host__ void setStartTime(double startTime);
__device__ __host__ double getMsGene(void *self);
__device__ __host__ double getOsSeqGene(void *self);
__device__ __host__ double getArrivT(void *self);
__device__ __host__ double getStartTime(void *self);
__device__ __host__ double getEndTime(void *self);
__device__ __host__ unsigned int getMachineNo(void *self);



#endif
