#ifndef __POPULATION_H__
#define __POPULATION_H__
#include <vector>
#include <include/job.h>
#include <include/job_base.h>
#include <include/machine.h>

typedef struct round_t{
    int round_no;
    int AMOUNT_OF_JOBS;
    job_t * jobs; // sample;
    process_time_t  ** process_times;
    int * size_of_process_times;
}round_t;


struct population_t{
    unsigned int no;
    struct {
        int AMOUNT_OF_MACHINES;
        int AMOUNT_OF_CHROMOSOMES;
        int AMOUNT_OF_R_CHROMOSOMES;
        double EVOLUTION_RATE;
        double SELECTION_RATE;
        int GENERATIONS;
    }parameters;
    
    struct {
        std::vector<round_t> rounds;
        machine_t * machines;
        tool_t * tools;
        wire_t * wires;
    }samples;

    struct {
        job_t ** jobs;
        machine_t **machines;
        tool_t ** tools;
        wire_t ** wires;
    }device_objects;

    struct {
        job_t ** address_of_cujobs;
        machine_t ** address_of_cumachines;
        tool_t ** address_of_tools;
        wire_t ** address_of_wires;
        process_time_t **address_of_process_time_entry;
    }host_objects;

};

#endif
