#include "include/job_base.h"
#include <tests/include/test_job_base.h>

job_t * newJob(int sizeof_pt){
	job_t * j = (job_t*)malloc(sizeof(job_t));
	initJobBase(&j->base);
	job_base_operations_t jbops = JOB_BASE_OPS;
	jbops.setProcessTime(&j->base, NULL, sizeof_pt);
	return j;

}
