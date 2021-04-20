#ifndef __MACHINE_BASE_H__
#define __MACHINE_BASE_H__

#include "include/linked_list.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <include/job_base.h>

#ifndef MACHINE_BASE_OPS
#define MACHINE_BASE_OPS machine_base_operations_t{                    \
    .reset = resetMachineBase,                                         \
    .addJob = __addJob,                                                \
    .sortJob = __sortJob,                                              \
    .getSizeOfJobs = getSizeOfJobs,                                    \
}
#endif


typedef struct machine_base_t machine_base_t;

machine_base_t * newMachineBase(unsigned int machine_no);

struct machine_base_t{
	list_ele_t * root;
	list_ele_t * tail;
	unsigned int machine_no;
	unsigned int size_of_jobs;
	unsigned int avaliable_time;
};

struct machine_base_operations_t{
	void (*init)(void * self);

	void (*reset)(machine_base_t *self);

	void (*addJob)(machine_base_t *self, list_ele_t*);
	void (*sortJob)(machine_base_t *self, list_operations_t *ops);
	unsigned int (*getSizeOfJobs)(machine_base_t *self);
	void (*getQuality)(machine_base_t *self);
	
	size_t sizeof_setup_time_function_array;	
	double (*setUpTime[])(machine_base_t *self);
};


__device__ __host__ void resetMachineBase(machine_base_t *_self);
__device__ __host__ unsigned int getSizeOfJobs(machine_base_t* _self);

__device__ __host__ void initMachineBase(machine_base_t *_self);
__device__ __host__ void __addJob(machine_base_t *_self, list_ele_t *);
__device__ __host__ void __sortJob(machine_base_t *_self, list_operations_t *ops);

#endif
