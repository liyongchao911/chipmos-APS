#include "include/linked_list.h"
#include <include/job_base.h>
#include <numeric>




// constructor and initialization

__device__ __host__ void resetJobBase(job_base_t *self){
	self->start_time = 0;
	self->end_time = 0;
}


// setter
__device__ __host__ void setMsGenePointer(job_base_t *self, double * ms_gene){
	self->ms_gene = ms_gene;
}

__device__ __host__ void setOsSeqGenePointer(job_base_t *self, double * os_seq_gene){
	self->os_seq_gene = os_seq_gene;
}

__device__ __host__ void setProcessTime(job_base_t *self, process_time_t *ptime, unsigned int size_of_process_time){
	self->process_time = ptime;

	if(size_of_process_time != 0){
		self->size_of_process_time = size_of_process_time;
    	self->partition = 1.0 / (double)size_of_process_time;
	}
}

__device__ __host__ void setArrivT(job_base_t *self, double arriv_time){
    self->arriv_t = arriv_time;
}

__device__ __host__ void setStartTime(job_base_t *self, double start_time){
    self->start_time = start_time;
	if(self->process_time){
		self->end_time = self->start_time + self->process_time[self->machine_no].process_time;	
	}
}

// getter
__device__ __host__ double getMsGene(job_base_t *self){
	return *(self->ms_gene);
}

__device__ __host__ double getOsSeqGene(job_base_t *self){
    return *(self->os_seq_gene);
}

__device__ __host__ unsigned int getMachineNo(job_base_t *self){
    return self->machine_no;
}

__device__ __host__ double getArrivT(job_base_t *self){
    return self->arriv_t;
}

__device__ __host__ double getStartTime(job_base_t *self){
    return self->start_time;
}

__device__ __host__ double getEndTime(job_base_t *self){
    return self->end_time;
}

// operation
__device__ __host__ unsigned int machineSelection(job_base_t *self){
    //calculate which number of machine(from 1 to n) that corresponds to partition
	
	double ms_gene = getMsGene(self);
	unsigned int val = ms_gene / self->partition + 0.0001;
	if(self->process_time){
		self->machine_no = self->process_time[val].machine_no;
	}
	return val;
}

__device__ __host__ void initJobBase(void *_self){
	job_base_t * self = (job_base_t *)_self;
	self->ms_gene = self->os_seq_gene = NULL;
	self->process_time = NULL;
}


job_base_t * newJobBase(){
	job_base_t *jb = (job_base_t*)malloc(sizeof(job_base_t));
	initJobBase(jb);
	return jb;
}
