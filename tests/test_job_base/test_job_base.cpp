#include "test_job_base.h"

Job * newJob(int sizeof_pt){
	Job * j = (Job*)malloc(sizeof(Job));
	initJobBase(&j->base);
	j->base.setProcessTime(&j->base, NULL, sizeof_pt);
	return j;

}
