#include "include/job_base.h"
#include "include/linked_list.h"
#include <include/machine_base.h>


__device__ __host__ void reset(void *_self){
	MachineBase * self = (MachineBase *)_self;
	self->size_of_jobs = 0;
	self->root = self->tail = NULL;
}



__device__ __host__ void addJob(void *_self, JobBase * job)
{
	MachineBase *self =  (MachineBase *)_self;

	if (self->size_of_jobs == 0) {
		self->tail = self->root = job;	
	} else {
		self->tail->ele->setNext(self->tail->ele, job->ele); // add into the list
		job->ele->setPrev(job->ele, self->tail->ele); // connect to prev
		self->tail = job;	// move the tail
	}
	++self->size_of_jobs;
}

__device__ __host__ unsigned int getSizeOfJobs(void *_self)
{
	MachineBase *self = (MachineBase*)_self;
	return self->size_of_jobs;
}

__device__ __host__ void sortJob(void *_self)
{
	MachineBase *self = (MachineBase *)_self;
	LinkedListElement * ele = NULL;
	self->root->ele = linkedListMergeSort(self->root->ele);
	ele = self->root->ele;
	while(ele && ele->next){
		ele = (LinkedListElement*)ele->next;
	}
	self->tail = (JobBase *)ele->pDerivedObject;
	
}

__device__ __host__ void init(void *_self){
	MachineBase * self = (MachineBase *)_self;
	self->reset = reset;
	self->addJob = addJob;
	self->sortJob = sortJob;
	self->getSizeOfJobs = getSizeOfJobs;
	self->getSizeOfJobs = NULL;

	self->reset(self);
}

MachineBase * newMachineBase(unsigned int machine_no){
	MachineBase * mb = (MachineBase*)malloc(sizeof(MachineBase));
	mb->machine_no = machine_no;
	mb->init = init;

	mb->init(mb);	
	return mb;
}
