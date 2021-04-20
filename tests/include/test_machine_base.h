#ifndef __TEST_MACHINE_BASE_H__
#define __TEST_MACHINE_BASE_H__

#include "include/linked_list.h"
#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
#include <include/machine_base.h>
#include <include/job_base.h>
#include <include/common.h>
#include <cuda.h>

#ifdef MACHINE_BASE_OPS
#undef MACHINE_BASE_OPS
#endif

#define MACHINE_BASE_OPS machine_base_operations_t{\
	.init = initMachine, \
    .reset = resetMachineBase,\
	.addJob = __addJob,\
	.sortJob = __sortJob,\
	.getSizeOfJobs = getSizeOfJobs\
}

struct job_t{
	job_base_t base;
	list_ele_t ele;
	double val;
};

struct Machine{
	unsigned int machine_no;
	machine_base_t base;
};

__device__ __host__ double machineSortJobs(void * self);

__device__ __host__ double jobGetValue(void *);

__device__ __host__ void initMachine(void *self);

__device__ __host__ void initJob(job_t * _self);

job_t * newJob(double val);

Machine *newMachine();


#endif
