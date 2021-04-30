#ifndef __JOB_BASE_H__
#define __JOB_BASE_H__

#include <stddef.h>
#include <include/def.h>
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
typedef struct job_base_operations_t job_base_operations_t;

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
	unsigned int qty;
	unsigned int machine_no;
	double arriv_t;
	double start_time;
	double end_time;
};

struct job_base_operations_t{
	// constructor and initialization
	void (*init)(void *self);
	void (*reset)(job_base_t *self);
	
	// setter
	void (*set_ms_gene_addr)(job_base_t *self, double * ms_gene);
	void (*set_os_gene_addr)(job_base_t *self, double *os_seq_gene);
	void (*set_process_time)(job_base_t *self, process_time_t *, unsigned int size_of_process_time);
	void (*set_arrival_time)(job_base_t *self, double arrivT);
	void (*set_start_time)(job_base_t *self, double startTime);

	// getter
	double (*get_ms_gene)(job_base_t *self);
	double (*get_os_gene)(job_base_t *self);
	double (*get_arrival_time)(job_base_t *self);
	double (*get_start_time)(job_base_t *self);
	double (*get_end_time)(job_base_t *self);
	unsigned int (*get_machine_no)(job_base_t *self);

	// operation
	unsigned int (*machine_selection)(job_base_t *self);
};

job_base_t * job_base_new();
__qualifier__ void job_base_init(void *self);
__qualifier__ void job_base_reset(job_base_t *self);
__qualifier__ void set_ms_gene_addr(job_base_t *self, double *ms_gene);
__qualifier__ void set_os_gene_addr(job_base_t *self, double *os_seq_gene);
__qualifier__ void set_process_time(job_base_t *self, process_time_t * pt, unsigned int size_of_process_time);
__qualifier__ void set_arrival_time(job_base_t *self, double arrivT);
__qualifier__ void set_start_time(job_base_t *self, double startTime);
__qualifier__ double get_ms_gene(job_base_t *self);
__qualifier__ double get_os_gene(job_base_t *self);
__qualifier__ double get_arrival_time(job_base_t *self);
__qualifier__ double get_start_time(job_base_t *self);
__qualifier__ double get_end_time(job_base_t *self);
__qualifier__ unsigned int get_machine_no(job_base_t *self);
__qualifier__ unsigned int machine_selection(job_base_t *self);

#ifndef JOB_BASE_OPS
#define JOB_BASE_OPS job_base_operations_t{                  \
    .init                    = job_base_init,                \
	.reset                   = job_base_reset,               \
	.set_ms_gene_addr        = set_ms_gene_addr,             \
	.set_os_gene_addr        = set_os_gene_addr,             \
	.set_process_time        = set_process_time,             \
	.set_arrival_time        = set_arrival_time,             \
	.set_start_time          = set_start_time,               \
	.get_ms_gene             = get_ms_gene,                  \
	.get_os_gene             = get_os_gene,                  \
	.get_arrival_time        = get_arrival_time,             \
	.get_start_time          = get_start_time,               \
	.get_end_time            = get_end_time,                 \
	.get_machine_no          = get_machine_no,               \
	.machine_selection       = machine_selection             \
}
#endif

#endif
