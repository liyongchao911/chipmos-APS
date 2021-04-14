#ifndef __TEST_JOB_BASE_H__
#define __TEST_JOB_BASE_H__

#include <include/job_base.h>

typedef struct Job Job;

struct Job{
	JobBase base;
};

Job * newJob(int sizeof_pt);

#endif
