//
// Created by eugene on 2021/7/5.
//

#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#include "include/chromosome.h"
#include "include/entity.h"
#include "include/infra.h"
#include "include/lot.h"
#include "include/lots.h"
#include "include/population.h"
#include "include/route.h"

/**
 * initializeOperations() - initialize the operations of the population
 * machine_ops, job_ops and list_ops are initialized in this function.
 * machine_ops is a pointer to machine_base_operations_t which is a incomplete
 * type. The size of memory pointed by machine_ops is determined by
 * sizeof(machine_base_operations_t) and the number of setup time function times
 * sizeof(setup_time_t). In the function, the function pointers in the
 * machine_ops point to correct functions.
 *
 * list_ops is a pointer to list_operations_t. In the function, list_ops is
 * initialized by LINKED_LIST_OPS
 *
 * job_ops is a pointer to job_base_operations_t. In the function job_ops is
 * initialized by JOB_BASE_OPS
 * @param pop : a population whose operations need to be initialized.
 */
void initializeOperations(population_t *pop);


/**
 * createARound () - create a scheduling round
 * In the function, a round_t instance is generated and initialized. round_t
 * instance has the information about the scheduling problem such as which jobs
 * are considered in this round, the amount of jobs, the process time of each
 * job, the size of can_run_machines of each job and the resources list used in
 * this round.
 *
 * @param group : a vector of lot_group_t instance
 * @param machines : machines_t instance which includes all usable machines
 * @param tools : ancillary resources. heatblock
 * @param wires : ancillary resources.
 * @return a initialized round_t type instance.
 */
round_t createARound(std::vector<lot_group_t> group,
                     machines_t &machines,
                     ancillary_resources_t &tools,
                     ancillary_resources_t &wires);

/**
 * initializePopulation () - initialize a population
 * In this function, the round field of population_t is initialized by
 * createARound. In this function, the chromosomes are also initialized.
 * @param pop : a population
 * @param machines : machines_t instance which includes all usable machines
 * @param tools : ancillary resources. heatblock
 * @param wires : another ancillary resources.
 * @param round : which round
 */
void initializePopulation(population_t *pop,
                          machines_t &machines,
                          ancillary_resources_t &tools,
                          ancillary_resources_t &wires);



/**
 * geneticAlgorithm () - genetic algorithm does scheduling
 * Genetic algorithm is performed in this function. After doing scheduling, the
 * result is stored into result.csv file and the file opened mode is "a+" which
 * means that the result of different will be appended on the tail of same file
 * @param pop : a population which carries the algorithm objects and algorithm
 * parameters
 */
void geneticAlgorithm(population_t *pop);

/**
 * freeJobs () - free the jobs created in function createARound
 * @param round : a pointer pointing on a round_t instance
 */
void freeJobs(round_t *round);

/**
 * freeResources () - free the resources created in function createARound
 * The resources only include tools and wires. After freeing the array, the
 * tools and the wires fields are set to nullptr
 * @param round : a pointer pointing on a round_t instance
 */
void freeResources(round_t *round);

/**
 * freeChromosomes () - free the chromosomes which are allocated in
 * initializePopulation After freeing the chromosomes the chromosomes field of
 * population will be set to nullptr
 * @param pop : a population which is about to free the chromosomes.
 */
void freeChromosomes(population_t *pop);



void machineWriteBackToEntity(population_t *pop);

#endif
