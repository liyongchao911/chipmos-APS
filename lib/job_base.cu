#include <include/job_base.h>

// constructor and initialization
JobBase::JobBase(){
    init();
    partition = 1.0 / (double)size_of_process_time;
}

__device__ __host__ void JobBase::init(){
    this->start_time = 0;
    this->end_time = 0;
}

// setter
__device__ __host__ void JobBase::setMsGenePointer(double * ms_gene){
    this->ms_gene = ms_gene;
}

__device__ __host__ void JobBase::setOsSeqGenePointer(double * os_seq_gene){
    this->os_seq_gene = os_seq_gene;
}

__device__ __host__ void JobBase::setProcessTime(ProcessTime **ptime){
    this->process_time = ptime;
}

__device__ __host__ void JobBase::setArrivT(double arriv_time){
    this->arriv_t = arriv_time;
}

__device__ __host__ void JobBase::setStartTime(double start_time){
    this->start_time = start_time;
}

// getter
__device__ __host__ double JobBase::getMsGene(){
    return *ms_gene;
}

__device__ __host__ double JobBase::getOsSeqGene(){
    return *os_seq_gene;
}

__device__ __host__ unsigned int JobBase::getMachineNo(){
    return this->machine_no;
}

__device__ __host__ double JobBase::getArrivT(){
    return this->arriv_t;
}

__device__ __host__ double JobBase::getStartTime(){
    return this->start_time;
}

__device__ __host__ double JobBase::getEndTime(){
    return this->end_time;
}

// operation
__device__ __host__ unsigned int JobBase::machineSelection(){
    //calculate which number of machine(from 1 to n) that corresponds to partition
    unsigned int count = 0;
    if(*os_seq_gene == 0)
	    count = 1;
    while((this->partition * count) < *os_seq_gene){
        count++;
    }    
    return count;
}
