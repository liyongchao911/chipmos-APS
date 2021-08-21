#ifndef __MACHINE_H__
#define __MACHINE_H__

#include "include/infra.h"
#include "include/job.h"
#include "include/machine_base.h"
#include "include/parameters.h"

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

    info_t model_name;
    info_t location;
    ares_t *tool;
    ares_t *wire;
    job_t current_job;
    double makespan;
    double total_completion_time;
    double quality;
    int setup_times;
    void *ptr_derived_object;
} machine_t;


bool aresPtrComp(ares_t *a1, ares_t *a2);
bool aresComp(ares_t a1, ares_t a2);

void machineReset(machine_base_t *base);

double setupTimeCWN(job_base_t *_prev, job_base_t *_next, double time);
double setupTimeCK(job_base_t *_prev, job_base_t *_next, double time);
double setupTimeEU(job_base_t *_prev, job_base_t *_next, double time);
double setupTimeMC(job_base_t *_prev, job_base_t *_next, double time);
double setupTimeSC(job_base_t *_prev, job_base_t *_next, double time);
double setupTimeCSC(job_base_t *_prev, job_base_t *_next, double time);
double setupTimeUSC(job_base_t *_prev, job_base_t *_next, double time);

void scheduling(machine_t *mahcine,
                machine_base_operations_t *ops,
                weights_t weights,
                scheduling_parameters_t scheduling_parameters);

void staticAddJob(machine_t *machine,
                  job_t *job,
                  machine_base_operations_t *ops);

void insertAlgorithm(machine_t *machine,
                     machine_base_operations_t *ops,
                     weights_t weights,
                     scheduling_parameters_t scheduling_parameters);

void setLastJobInMachine(machine_t *machine);

void setJob2Scheduled(machine_t *machine);

#endif
