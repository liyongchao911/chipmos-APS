#include <tests/include/test_job_base.h>

job_t * newJob(int sizeof_pt){
	job_t * j = (job_t*)malloc(sizeof(job_t));
	initJobBase(&j->base);
	j->base.setProcessTime(&j->base, NULL, sizeof_pt);
	return j;

}
