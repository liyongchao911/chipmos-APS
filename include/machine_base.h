#ifndef __MACHINE_BASE_H__
#define __MACHINE_BASE_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <include/job_base.h>

typedef struct MachineBase MachineBase;

MachineBase * newMachineBase(unsigned int machine_no);

struct MachineBase{
	JobBase * root;
	JobBase * tail;
	unsigned int machine_no;
	unsigned int size_of_jobs;
	
	void (*init)(void *self);
	void (*reset)(void *self);

	void (*addJob)(void *self, JobBase *);
	void (*sortJob)(void *self);
	unsigned int (*getSizeOfJobs)(void *self);
	void (*getQuality)(void *self);
			
};


#endif
