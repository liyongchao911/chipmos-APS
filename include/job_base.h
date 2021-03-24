#ifndef __JOB_BASE_H__
#define __JOB_BASE_H__
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <include/linked_list.h>

/**
 * @brief Store process time and its corresponding machine number.
 *
 */
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


class JobBase : public LinkedList{
friend class TestJobBase;
protected:
	// genes point to chromosome's gene
	// use double const * type to prevent set the wrong value on gene
	double const * ms_gene;
	double const * os_seq_gene;
	
	// partition is the partition value of roulette.
	// for example : size of can run tools is 10, partition is 1/10
	double partition; 
	
	// process time
	// process_time is an 1-D array
	ProcessTime ** process_time; 
	unsigned int size_of_process_time;
	
	// job information
	unsigned int job_no;
	unsigned int machine_no;
	double arriv_t;
	double start_time;
	double end_time;
public:
	// constructor and initialization
	JobBase();
	__device__ __host__ void init();
	
	// setter
	__device__ __host__ void setMsGenePointer(double * ms_gene);
	__device__ __host__ void setOsSeqGenePointer(double * os_seq_gene);
	__device__ __host__ void setProcessTime(ProcessTime **, unsigned int size_of_process_time);
	__device__ __host__ void setArrivT(double);
    __device__ __host__ void setStartTime(double);

	// getter
	__device__ __host__ double getMsGene();
	__device__ __host__ double getOsSeqGene();
	__device__ __host__ unsigned int getMachineNo();
	__device__ __host__ double getArrivT();
	__device__ __host__ double getStartTime();
	__device__ __host__ double getEndTime();
	
	
	// operation
	__device__ __host__ unsigned int machineSelection();

};

#endif
