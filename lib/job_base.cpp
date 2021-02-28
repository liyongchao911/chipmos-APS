
#include "job_base.h"
#include<string>

using namespace std;


class ProcessTime{
private:
	unsigned int machine_no;
	double process_time;
public:
	ProcessTime(unsigned int machine_no, double process_time);

	__device__ __host__ void setProcessTime(double time);	
	__device__ __host__ unsigned int getMachineNo();
	__device__ __host__ double getProcessTime();
};



ProcessTime::ProcessTime(unsigned int machine_no, double process_time){
    this->machine_no = machine_no;
    this->process_time = process_time;
}

__device__ __host__ void ProcessTime::setProcessTime(double time){

}

__device__ __host__ unsigned int ProcessTime::getMachineNo(){
return machine_no;
}

__device__ __host__ double ProcessTime::getProcessTime(){

}

class Account { 
private:
    string id; 

};