#ifndef __MACHINE_BASE_H__
#define __MACHINE_BASE_H__

#include "include/linked_list.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <include/job_base.h>

typedef struct MachineBase MachineBase;

MachineBase * newMachineBase(unsigned int machine_no);

struct MachineBase{
	LinkedListElement * root;
	LinkedListElement * tail;
	unsigned int machine_no;
	unsigned int size_of_jobs;
	unsigned int avaliable_time;
	
	void (*init)(void *self);
	void (*reset)(void *self);

	void (*addJob)(void *self, void *job);
	void (*sortJob)(void *self);

	void (*__addJob)(void *self, LinkedListElement *);
	void (*__sortJob)(void *self);
	unsigned int (*getSizeOfJobs)(void *self);
	void (*getQuality)(void *self);
};

__device__ __host__ void resetMachineBase(void *_self);
__device__ __host__ unsigned int getSizeOfJobs(void* _self);

__device__ __host__ void initMachineBase(void *_self);
__device__ __host__ void __addJob(void *_self, LinkedListElement *);
__device__ __host__ void __sortJob(void *_self);

#endif
