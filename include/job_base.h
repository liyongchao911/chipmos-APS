#ifndef __JOB_BASE_H__
#define __JOB_BASE_H__
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <include/linked_list.h>



/**
 * @brief Store process time and its corresponding machine number.
 */
typedef struct process_time_t process_time_t;
struct process_time_t{
	unsigned int machine_no;
	double process_time;
	void *ptr_derived_object;
};

typedef struct job_base_t job_base_t;
struct job_base_t{
	void * ptr_derived_object;

	// genes point to chromosome's gene
	// use double const * type to prevent set the wrong value on gene
	double const * ms_gene;
	double const * os_seq_gene;
	
	// partition is the partkkkition value of roulette.
	// for example : size of can run tools is 10, partition is 1/10
	double partition; 
	
	// process time
	// process_time is an 1-D array
	process_time_t * process_time;
	unsigned int size_of_process_time;
	
	// job information
	unsigned int job_no;
	unsigned int machine_no;
	double arriv_t;
	double start_time;
	double end_time;
};

typedef struct job_base_operations_t{
	// constructor and initialization
	void (*init)(void *self);
	void (*reset)(job_base_t *self);
	
	// setter
	void (*setMsGenePointer)(job_base_t *self, double * ms_gene);
	void (*setOsSeqGenePointer)(job_base_t *self, double *os_seq_gene);
	void (*setProcessTime)(job_base_t *self, process_time_t *, unsigned int size_of_process_time);
	void (*setArrivT)(job_base_t *self, double arrivT);
	void (*setStartTime)(job_base_t *self, double startTime);

	// getter
	double (*getMsGene)(job_base_t *self);
	double (*getOsSeqGene)(job_base_t *self);
	double (*getArrivT)(job_base_t *self);
	double (*getStartTime)(job_base_t *self);
	double (*getEndTime)(job_base_t *self);
	unsigned int (*getMachineNo)(job_base_t *self);

	// operation
	unsigned int (*machineSelection)(job_base_t *self);
} job_base_operations_t;

job_base_t * newJobBase();
__device__ __host__ void initJobBase(void *self);
__device__ __host__ void resetJobBase(job_base_t *self);
__device__ __host__ void setMsGenePointer(job_base_t *self, double *ms_gene);
__device__ __host__ void setOsSeqGenePointer(job_base_t *self, double *os_seq_gene);
__device__ __host__ void setProcessTime(job_base_t *self, process_time_t * pt, unsigned int size_of_process_time);
__device__ __host__ void setArrivT(job_base_t *self, double arrivT);
__device__ __host__ void setStartTime(job_base_t *self, double startTime);
__device__ __host__ double getMsGene(job_base_t *self);
__device__ __host__ double getOsSeqGene(job_base_t *self);
__device__ __host__ double getArrivT(job_base_t *self);
__device__ __host__ double getStartTime(job_base_t *self);
__device__ __host__ double getEndTime(job_base_t *self);
__device__ __host__ unsigned int getMachineNo(job_base_t *self);
__device__ __host__ unsigned int machineSelection(job_base_t *self);

#ifndef JOB_BASE_OPS
#define JOB_BASE_OPS job_base_operations_t{              \
    .init                = initJobBase,                  \
	.reset               = resetJobBase,                 \
	.setMsGenePointer    = setMsGenePointer,             \
	.setOsSeqGenePointer = setOsSeqGenePointer,          \
	.setProcessTime      = setProcessTime,               \
	.setArrivT           = setArrivT,                    \
	.setStartTime        = setStartTime,                 \
	.getMsGene           = getMsGene,                    \
	.getOsSeqGene        = getOsSeqGene,                 \
	.getArrivT           = getArrivT,                    \
	.getStartTime        = getStartTime,                 \
	.getEndTime          = getEndTime,                   \
	.getMachineNo        = getMachineNo,                 \
	.machineSelection    = machineSelection              \
}
#endif

#endif
