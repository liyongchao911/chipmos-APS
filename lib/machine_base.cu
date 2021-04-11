#include "include/job_base.h"
#include "include/linked_list.h"
#include <include/machine_base.h>


__device__ __host__ void resetMachineBase(void *_self){
	MachineBase * self = (MachineBase *)_self;
	self->size_of_jobs = 0;
	self->root = self->tail = NULL;
}


__device__ __host__ void __addJob(void *_self, LinkedListElement * job)
{
	MachineBase *self =  (MachineBase *)_self;
	LinkedListElementOperation ops = LINKED_LIST_OPS();
	if (self->size_of_jobs == 0) {
		self->tail = self->root = job;
	} else {
		ops.setNext(self->tail, job);	
		// self->tail->setNext(self->tail, job); // add into the list
		self->tail = job;	// move the tail
	}
	++self->size_of_jobs;
}

__device__ __host__ void __sortJob(void *_self, LinkedListElementOperation *ops)
{
	MachineBase *self = (MachineBase *)_self;
	LinkedListElement * ele = NULL;
	// LINKED_LIST_OPS();
	// LinkedListElementOperation ops = LINKED_LIST_OPS();
	self->root = linkedListMergeSort(self->root, ops);
	ele = self->root;
	while(ele && ele->next){
		ele = (LinkedListElement*)ele->next;
	}
	self->tail = ele;
	
}


__device__ __host__ unsigned int getSizeOfJobs(void *_self)
{
	MachineBase *self = (MachineBase*)_self;
	return self->size_of_jobs;
}


__device__ __host__ void initMachineBase(void *_self){
	MachineBase * self = (MachineBase *)_self;
	self->reset = resetMachineBase;
	self->__addJob = __addJob;
	self->__sortJob = __sortJob;
	self->getSizeOfJobs = getSizeOfJobs;
	self->addJob = NULL;
	self->sortJob = NULL;
	self->getQuality = NULL;


	self->reset(self);
}

MachineBase * newMachineBase(unsigned int machine_no){
	MachineBase * mb = (MachineBase*)malloc(sizeof(MachineBase));
	mb->machine_no = machine_no;
	mb->init = initMachineBase;

	mb->init(mb);	
	return mb;
}
