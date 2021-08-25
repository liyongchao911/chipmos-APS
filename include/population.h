#ifndef __POPULATION_H__
#define __POPULATION_H__
#include <map>
#include <vector>
#include "include/chromosome_base.h"
#include "include/common.h"
#include "include/job.h"
#include "include/job_base.h"
#include "include/lot.h"
#include "include/machine.h"
#include "include/machine_base.h"
#include "include/parameters.h"


struct population_t {
    unsigned int no;
    struct {
        int AMOUNT_OF_CHROMOSOMES;
        int AMOUNT_OF_R_CHROMOSOMES;
        double EVOLUTION_RATE;
        double SELECTION_RATE;
        int GENERATIONS;
        int MAX_SETUP_TIMES;
        weights_t weights;
        scheduling_parameters_t scheduling_parameters;
    } parameters;

    struct {
        list_operations_t *list_ops;
        job_base_operations_t *job_ops;
        machine_base_operations_t *machine_ops;
    } operations;

    struct {
        job_t **jobs;
        machine_t **machines;
        int NUMBER_OF_JOBS;
        int NUMBER_OF_MACHINES;
    } objects;

    chromosome_base_t *chromosomes;
};

#endif
