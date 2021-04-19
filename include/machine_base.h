#ifndef __MACHINE_BASE_H__
#define __MACHINE_BASE_H__

#include "include/linked_list.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <include/job_base.h>

typedef struct machine_base_t machine_base_t;

machine_base_t * newMachineBase(unsigned int machine_no);

struct machine_base_t{
	list_ele_t * root;
	list_ele_t * tail;
	unsigned int machine_no;
	unsigned int size_of_jobs;
	unsigned int avaliable_time;
	
	void (*init)(void *self);
	void (*reset)(void *self);

	void (*addJob)(void *self, void *job);
	void (*sortJob)(void *self, list_operations_t *ops);

	void (*__addJob)(void *self, list_ele_t*);
	void (*__sortJob)(void *self, list_operations_t *ops);
	unsigned int (*getSizeOfJobs)(void *self);
	void (*getQuality)(void *self);

	// double (*setUpTime[])(void *self);
};

struct machine_base_operations_t{
	void (*init)(void *self);

	void (*reset)(void *self);
	void (*addJob)(void *self, void *job);
	void (*sortJob)(void *self, list_operations_t *ops);

	void (*__addJob)(void *self, list_ele_t*);
	void (*__sortJob)(void *self, list_operations_t *ops);
	unsigned int (*getSizeOfJobs)(void *self);
	void (*getQuality)(void *self);
	
	size_t sizeof_setup_time_function_array;	
	double (*setUpTime[])(void *self);
};


__device__ __host__ void resetMachineBase(void *_self);
__device__ __host__ unsigned int getSizeOfJobs(void* _self);

__device__ __host__ void initMachineBase(void *_self);
__device__ __host__ void __addJob(void *_self, list_ele_t *);
__device__ __host__ void __sortJob(void *_self, list_operations_t *ops);

#endif
