#ifndef __MACHINE_BASE_H__
#define __MACHINE_BASE_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <include/job_base.h>

class MachineBase{
protected:
	JobBase * root;
	JobBase * tail;
	unsigned int machine_no;
	unsigned int size_of_jobs;

	__device__ __host__ virtual void sortJobAlgorithm(JobBase *);
public:
			
	MachineBase(unsigned int machine_no);
	__device__ __host__ void init();
	
	__device__ __host__ void addJob(JobBase *);
	__device__ __host__ void sortJob();
	__device__ __host__ unsigned int getSizeOfJobs();
	
	__device__ __host__ virtual double getQuality()=0;

};


#endif
