#include <include/job_base.h>

// constructor and initialization
JobBase::JobBase(){

}

__device__ __host__ void JobBase::init(){
    this->arriv_t = 0;
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

__device__ __host__ void JobBase::setProcessTime(ProcessTime **){

}

// getter
__device__ __host__ double JobBase::getMsGene(){
    return this->ms_gene;
}

__device__ __host__ double JobBase::getOsSeqGene(){
    return this->os_seq_gene;
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

}
