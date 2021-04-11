#include <include/machine_base.h>
#include <include/job_base.h>
#include <include/linked_list.h>
#include "test_machine_base.h"

#define MAX_MACHINE_SIZE 10000
#define MAX_JOB_SIZE 20000

__device__ __host__ double jobGetValue(void *_self){
	LinkedListElement * self = (LinkedListElement *)_self;
	Job * j = (Job *)self->ptr_derived_object;
	return j->val;
}

__device__ __host__ void initJob(Job *self){
	initList(&self->ele);
	self->ele.ptr_derived_object = self;
	self->ele.getValue = jobGetValue;

	initJobBase(&self->base);
	self->base.ptr_derived_object = self;
}

Job * newJob(double val){
	Job * j = (Job *)malloc(sizeof(Job));
	initList(&j->ele);
	initJobBase(&j->base);
	initJob(j);
	j->val = val;
	return j;
}

__device__ __host__ void addJob(void *_self, void *_job){
	MachineBase *self = (MachineBase*)_self;
	Job * job = (Job*)_job;
	self->__addJob(self, &job->ele);
}

__device__ __host__ void sortJob(void *_self, LinkedListElementOperation *ops){
	Machine *self = (Machine*)_self;
	self->base.__sortJob(&self->base, ops);	
}

__device__ __host__ void initMachine(void *_self){
	Machine *self = (Machine*)_self;
	initMachineBase(&self->base);
	self->base.addJob = addJob;
	self->base.sortJob = sortJob;
	self->base.__addJob = __addJob;
	self->base.__sortJob = __sortJob;
}

Machine * newMachine(){
	Machine * m = (Machine*)malloc(sizeof(Machine));
	initMachine(m);
	return m;
}

int cmpint(const void *a, const void *b){
	return *(int*)a > *(int*)b;
}


