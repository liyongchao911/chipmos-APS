#include "include/job_base.h"
#include "include/linked_list.h"
#include <include/machine_base.h>


__device__ __host__ void
resetMachineBase(machine_base_t *self)
{
	self->size_of_jobs = 0;
	self->root = self->tail = NULL;
}


__device__ __host__ void 
__addJob(
		machine_base_t *self, 
		list_ele_t * job
)
{
	job->next = job->prev = NULL;
	list_operations_t ops = LINKED_LIST_OPS;
	if (self->size_of_jobs == 0) {
		self->tail = self->root = job;
	} else {
		ops.setNext(self->tail, job);	
		// self->tail->setNext(self->tail, job); // add into the list
		self->tail = job;	// move the tail
	}
	++self->size_of_jobs;
}

__device__ __host__ void 
__sortJob(
		machine_base_t *self, 
		list_operations_t *ops
)
{
	if(self->size_of_jobs == 0){
		return;
	}
	list_ele_t * ele = NULL;
	self->root = linkedListMergeSort(self->root, ops);
	ele = self->root;
	while(ele && ele->next){
		ele = (list_ele_t*)ele->next;
	}
	self->tail = ele;
}


__device__ __host__ unsigned int 
getSizeOfJobs(machine_base_t *self)
{
	return self->size_of_jobs;
}


__device__ __host__ void 
initMachineBase(machine_base_t *self)
{
	resetMachineBase(self);
}

machine_base_t * 
newMachineBase(unsigned int machine_no)
{
	machine_base_t * mb = (machine_base_t*)malloc(sizeof(machine_base_t));
	mb->machine_no = machine_no;
	return mb;
}
