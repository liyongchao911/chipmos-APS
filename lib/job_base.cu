#include "include/linked_list.h"
#include <include/job_base.h>




// constructor and initialization

__device__ __host__ void resetJobBase(void *_self){
	job_base_t *self = (job_base_t*)_self;
	self->start_time = 0;
	self->end_time = 0;
}


// setter
__device__ __host__ void setMsGenePointer(void *_self, double * ms_gene){
	job_base_t *self = (job_base_t *)_self;
	self->ms_gene = ms_gene;
}

__device__ __host__ void setOsSeqGenePointer(void *_self, double * os_seq_gene){
	job_base_t *self = (job_base_t *)_self;
	self->os_seq_gene = os_seq_gene;
}

__device__ __host__ void setProcessTime(void *_self, process_time_t *ptime, unsigned int size_of_process_time){
	job_base_t *self = (job_base_t *)_self;
	self->process_time = ptime;

	if(size_of_process_time != 0){
		self->size_of_process_time = size_of_process_time;
    	self->partition = 1.0 / (double)size_of_process_time;
	}
}

__device__ __host__ void setArrivT(void *_self, double arriv_time){
	job_base_t *self = (job_base_t*)_self;
    self->arriv_t = arriv_time;
}

__device__ __host__ void setStartTime(void *_self, double start_time){
	job_base_t *self = (job_base_t*)_self;
    self->start_time = start_time;
}

// getter
__device__ __host__ double getMsGene(void *_self){
	job_base_t *self = (job_base_t*)_self;
	return *(self->ms_gene);
}

__device__ __host__ double getOsSeqGene(void *_self){
	job_base_t *self = (job_base_t*)_self;
    return *(self->os_seq_gene);
}

__device__ __host__ unsigned int getMachineNo(void *_self){
	job_base_t *self = (job_base_t*)_self;
    return self->machine_no;
}

__device__ __host__ double getArrivT(void *_self){
	job_base_t *self = (job_base_t*)_self;
    return self->arriv_t;
}

__device__ __host__ double getStartTime(void *_self){
	job_base_t *self = (job_base_t*)_self;
    return self->start_time;
}

__device__ __host__ double getEndTime(void *_self){
	job_base_t *self = (job_base_t*)_self;
    return self->end_time;
}

// operation
__device__ __host__ unsigned int machineSelection(void *_self){
    //calculate which number of machine(from 1 to n) that corresponds to partition
	job_base_t *self = (job_base_t *)_self;
    unsigned int count = 0;
    double cal_partition = 0.0;
	double ms_gene = self->getMsGene(self);

	double partition = self->partition;
    if(ms_gene == 0)
	    count = 1;
    while(cal_partition < ms_gene){
        cal_partition += partition;
        count++;
    }    
    return count;
}

__device__ __host__ void initJobBase(void *_self){
	job_base_t * self = (job_base_t *)_self;
	self->ms_gene = self->os_seq_gene = NULL;
	self->reset = resetJobBase;
	self->setMsGenePointer = setMsGenePointer;
	self->setOsSeqGenePointer = setOsSeqGenePointer;
	self->setProcessTime = setProcessTime;
	self->setArrivT = setArrivT;
	self->setStartTime = setStartTime;

	self->getMsGene = getMsGene;
	self->getOsSeqGene = getOsSeqGene;
	self->getArrivT = getArrivT;
	self->getStartTime = getStartTime;
	self->getEndTime = getEndTime;
	self->getMachineNo = getMachineNo;

	self->machineSelection = machineSelection;

	self->reset(self);
}


job_base_t * newJobBase(){
	job_base_t *jb = (job_base_t*)malloc(sizeof(job_base_t));

	// jb->ele = newLinkedListElement();
	// jb->ele->pDerivedObject = jb;
	// jb->ele->getValue = getOsSeqGene; // virtual ^_^

	jb->init = initJobBase;

	jb->init(jb);	
	return jb;
}
