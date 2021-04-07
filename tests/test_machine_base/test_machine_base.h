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

struct Job{
	JobBase base;
	LinkedListElement ele;
	double val;
};

struct Machine{
	MachineBase base;
};

__device__ __host__ double machineSortJobs(void * self);

__device__ __host__ double jobGetValue(void *);

__device__ __host__ void addJob(void *_self, void * job);

__device__ __host__ void sortJob(void *_self);

__device__ __host__ void initMachine(void *self);

__device__ __host__ void initJob(Job * _self);

Job * newJob(double val);

Machine *newMachine();


#endif
