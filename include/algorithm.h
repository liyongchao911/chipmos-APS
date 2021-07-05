//
// Created by eugene on 2021/7/5.
//

#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#include "include/entity.h"
#include "include/infra.h"
#include "include/lot.h"
#include "include/lots.h"
#include "include/population.h"
#include "include/route.h"
#include "include/chromosome.h"

round_t createARound(std::vector<lot_group_t> group,
                     machines_t &machines,
                     ancillary_resources_t &tools,
                     ancillary_resources_t &wires);

void initializePopulation(population_t *pop,
                          machines_t &machines,
                          ancillary_resources_t &tools,
                          ancillary_resources_t &wires,
                          int round);

void initializeOperations(population_t *pop);


void geneticAlgorithm(population_t *pop);

void freeJobs(round_t *round);
void freeResources(round_t * round);
void freeChromosomes(population_t *pop);

#endif
