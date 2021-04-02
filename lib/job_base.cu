#include "include/linked_list.h"
#include <include/job_base.h>

// void init(void *self);
// void reset(void *self);
// 
// void setMsGenePointer(void *self, double *ms_gene);
// void setOsSeqGenePointer(void *self, double *os_seq_gene);
// void setProcessTime(void *self, ProcessTime ** pt, unsigned int size_of_process_time);
// void setArrivT(void *self, double arrivT);
// void setStartTime(double startTime);
// 
// double getMsGene(void *self);
// double getOsSeqGene(void *self);
// double getArrivT(void *self);
// double getStartTime(void *self);
// double getEndTime(void *self);
// unsigned int getMachineNo(void *self);
// 




// constructor and initialization

void resetJobBase(void *_self){
	JobBase *self = (JobBase*)_self;
	self->start_time = 0;
	self->end_time = 0;
}


// setter
__device__ __host__ void setMsGenePointer(void *_self, double * ms_gene){
	JobBase *self = (JobBase *)_self;
	self->ms_gene = ms_gene;
}

__device__ __host__ void setOsSeqGenePointer(void *_self, double * os_seq_gene){
	JobBase *self = (JobBase *)_self;
	self->os_seq_gene = os_seq_gene;
}

__device__ __host__ void setProcessTime(void *_self, ProcessTime **ptime, unsigned int size_of_process_time){
	JobBase *self = (JobBase *)_self; 
	self->process_time = ptime;

	if(size_of_process_time != 0){
		self->size_of_process_time = size_of_process_time;
    	self->partition = 1.0 / (double)size_of_process_time;
	}
}

__device__ __host__ void setArrivT(void *_self, double arriv_time){
	JobBase *self = (JobBase*)_self;
    self->arriv_t = arriv_time;
}

__device__ __host__ void setStartTime(void *_self, double start_time){
	JobBase *self = (JobBase*)_self;
    self->start_time = start_time;
}

// getter
__device__ __host__ double getMsGene(void *_self){
	JobBase *self = (JobBase*)_self;
	return *(self->ms_gene);
}

__device__ __host__ double getOsSeqGene(void *_self){
	JobBase *self = (JobBase*)_self;
    return *(self->os_seq_gene);
}

__device__ __host__ unsigned int getMachineNo(void *_self){
	JobBase *self = (JobBase*)_self;
    return self->machine_no;
}

__device__ __host__ double getArrivT(void *_self){
	JobBase *self = (JobBase*)_self;
    return self->arriv_t;
}

__device__ __host__ double getStartTime(void *_self){
	JobBase *self = (JobBase*)_self;
    return self->start_time;
}

__device__ __host__ double getEndTime(void *_self){
	JobBase *self = (JobBase*)_self;
    return self->end_time;
}

// operation
__device__ __host__ unsigned int machineSelection(void *_self){
    //calculate which number of machine(from 1 to n) that corresponds to partition
	JobBase *self = (JobBase *)_self;
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
	JobBase * self = (JobBase *)_self;
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


JobBase * newJobBase(){
	JobBase *jb = (JobBase*)malloc(sizeof(JobBase));

	jb->ele = newLinkedListElement();
	jb->ele->pDerivedObject = jb;
	jb->ele->getValue = getOsSeqGene; // virtual ^_^

	jb->init = initJobBase;

	jb->init(jb);	
	return jb;
}
