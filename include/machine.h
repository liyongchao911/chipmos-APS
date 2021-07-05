#ifndef __MACHINE_H__
#define __MACHINE_H__

#include <include/infra.h>
#include <include/job.h>
#include <include/machine_base.h>

typedef struct __info_t machine_info_t;
typedef struct __info_t tool_info_t;
typedef struct __info_t wire_info_t;

typedef struct ancillary_resource_t {
    unsigned int no;
    struct __info_t name;
    unsigned int machine_no;
    double time;
} ares_t;

typedef ares_t tool_t;
typedef ares_t wire_t;


typedef struct __machine_t {
    machine_base_t base;
    ares_t *tool;
    ares_t *wire;
    job_t current_job;
    double makespan;
    double total_completion_time;
} machine_t;

bool aresPtrComp(ares_t *a1, ares_t *a2);
bool aresComp(ares_t a1, ares_ta2);

void machineReset(machine_base_t *base);

double setupTimeCWN(job_base_t *_prev, job_base_t *_next);
double setupTimeCK(job_base_t *_prev, job_base_t *_next);
double setupTimeEU(job_base_t *_prev, job_base_t *_next);
double setupTimeMCSC(job_base_t *_prev, job_base_t *_next);
double setupTimeCSC(job_base_t *_prev, job_base_t *_next);
double setupTimeUSC(job_base_t *_prev, job_base_t *_next);

void scheduling(machine_t *mahcine, machine_base_operations_t *ops);

void setLastJobInMachine(machine_t *machine);

#endif
