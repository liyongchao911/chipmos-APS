#ifndef __CHROMOSOME_H__
#define __CHROMOSOME_H__

#include <include/chromosome_base.h>
#include <include/machine.h>
#include <include/infra.h>
#include <string.h>
#include <map>

typedef struct chromosome_linker{
    chromosome_base_t chromosome;
    double value;
}chromosome_linker_t;

void crossover(chromosome_base_t p1, chromosome_base_t p2, chromosome_base_t c1, chromosome_base_t c2);
void mutation(chromosome_base_t p, chromosome_base_t c);
void copyChromosome(chromosome_base_t c1, chromosome_base_t c2);

double decoding(chromosome_base_t chromosome, job_t * jobs, std::map<unsigned int, machine_t *>machines, machine_base_operations_t *machine_ops, list_operations_t *list_ops, int AMOUNT_OF_JOBS);

#endif
