#ifndef __MACHINE_BASE_H__
#define __MACHINE_BASE_H__

#include <stddef.h>
#include <include/def.h>
#include <include/linked_list.h>
#include <include/job_base.h>

#ifndef MACHINE_BASE_OPS
#define MACHINE_BASE_OPS machine_base_operations_t{                    \
    .reset = machine_base_reset,                                         \
    .add_job = _machine_base_add_job,                                                \
    .sort_job = _machine_base_sort_job,                                              \
    .get_size_of_jobs = machine_base_get_size_of_jobs,                                    \
}
#endif


typedef struct machine_base_t machine_base_t;

machine_base_t * machine_base_new(unsigned int machine_no);

struct machine_base_t{
	list_ele_t * root;
	list_ele_t * tail;
	unsigned int machine_no;
	unsigned int size_of_jobs;
	unsigned int avaliable_time;
};

struct machine_base_operations_t{
	void (*init)(void * self);

	void (*reset)(machine_base_t *self);

	void (*add_job)(machine_base_t *self, list_ele_t*);
	void (*sort_job)(machine_base_t *self, list_operations_t *ops);
	unsigned int (*get_size_of_jobs)(machine_base_t *self);
	void (*get_quality)(machine_base_t *self);
	
	size_t sizeof_setup_time_function_array;	
	double (*set_up_times[])(machine_base_t *self);
};


__qualifier__ void machine_base_reset(machine_base_t *_self);
__qualifier__ unsigned int machine_base_get_size_of_jobs(machine_base_t* _self);

__qualifier__ void machine_base_init(machine_base_t *_self);
__qualifier__ void _machine_base_add_job(machine_base_t *_self, list_ele_t *);
__qualifier__ void _machine_base_sort_job(machine_base_t *_self, list_operations_t *ops);

#endif
