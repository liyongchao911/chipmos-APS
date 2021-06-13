#ifndef __MACHINE_H__
#define __MACHINE_H__

#include <include/job.h>
#include <include/machine_base.h>
#include <include/infra.h>

typedef struct __info_t machine_info_t;
typedef struct __info_t tool_info_t;
typedef struct __info_t wire_info_t;

// typedef struct __tool_t{
//     unsigned int no;
//     tool_info_t name;
//     tool_info_t entity_name;
//     time_t time;
// }tool_t;
// 
// typedef struct __wire_t{
//     unsigned int no;
//     wire_info_t name;
//     wire_info_t entity_name;
//     time_t time;
// }wire_t;



typedef struct ancillary_resource_t{
    unsigned int no;
    struct __info_t name;
    unsigned int machine_no;
    double time;
}ares_t;

typedef ares_t tool_t;
typedef ares_t wire_t;


typedef struct __machine_t{
    machine_base_t base;
    ares_t * tool;
    ares_t * wire;
    double makespan;
}machine_t;

bool ares_ptr_comp(ares_t *, ares_t *);
bool ares_comp(ares_t , ares_t);

void machine_reset(machine_base_t * base);

double setup_time_CWN(job_base_t *, job_base_t*);
double setup_time_CK(job_base_t*, job_base_t *);
double setup_time_EU(job_base_t*, job_base_t *);
double setup_time_MC_SC(job_base_t*, job_base_t *);
double setup_time_CSC(job_base_t*, job_base_t *);
double setup_time_USC(job_base_t*, job_base_t *);

void scheduling(machine_t *mahcine, machine_base_operations_t *ops);

#endif
