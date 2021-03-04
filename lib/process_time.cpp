
#include <include/job_base.h>
#include <string>

using namespace std;


ProcessTime::ProcessTime(unsigned int machine_no, double process_time){
    this->machine_no = machine_no;
    this->process_time = process_time;
}

__device__ __host__ void ProcessTime::setProcessTime(double time){
	this->process_time = time;
}

__device__ __host__ unsigned int ProcessTime::getMachineNo(){
    return machine_no;
}

__device__ __host__ double ProcessTime::getProcessTime(){
    return process_time;
}


