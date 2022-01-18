#ifndef __CHROMOSOME_H__
#define __CHROMOSOME_H__
#include <cstring>
#include <map>

#include "include/chromosome_base.h"
#include "include/infra.h"
#include "include/job_base.h"
#include "include/machine.h"
#include "include/parameters.h"

/**
 * struct chromosome_linker : linker is used to link the chromosome_base_t
 * object and its fitness value chromosome_linker is used in genetic algorithm
 * to store which chromosomes should do crossover or mutation.
 */
typedef struct chromosome_linker {
    chromosome_base_t chromosome;
    double value;  // chromosome's fitness value
} chromosome_linker_t;

/**
 * crossover () - crossover for 2 chromosomes
 * In the function, 2 chromosome, p1 and p2, exchange a segment of genes and
 * produce 2 genes, c1 and c2.
 * @param p1 : parent1
 * @param p2 : parent2
 * @param c1 : offspring 1
 * @param c2 : offspring 2
 */
void crossover(chromosome_base_t p1,
               chromosome_base_t p2,
               chromosome_base_t c1,
               chromosome_base_t c2);

/**
 * mutation () - mutation for a chromosome
 * In the function, 1 chromosome, p, does mutation, and store the result to
 * offspring c. The mutation method is randomly choose a gene and change the
 * value of gene by another random number in the range (0, 1).
 * @param p : parent1
 * @param c : offspring
 */
void mutation(chromosome_base_t p, chromosome_base_t c);

/**
 * copyChromosome () - copy the genes of chromosomes from src to dest
 * @param src
 * @param dest
 */
void copyChromosome(chromosome_base_t dest, chromosome_base_t src);

/**
 * decoding () - decode a chromosome
 * The first step of decoding is reset the machines. The reset function is set
 * in machine_ops. The second step is machine selection. The third step is
 * sorting the job. The fourth step is scheduling. The function finally returns
 * the fitness value.
 * @param chromosome : the chromosome is about to decode
 * @param jobs : job array
 * @param machines : a map container which map unsigned to machine_t *
 * @param machine_ops : machine operations
 * @param list_ops : linked list operations
 * @param job_ops : job operation
 * @param NUMBER_OF_JOBS : amount of jobs
 * @param MAX_SETUP_TIMES : maximum of setup times
 * @return the fitness value of chromosome
 */
double decoding(chromosome_base_t chromosome,
                job_t **jobs,
                machine_t **machines,
                machine_base_operations_t *machine_ops,
                list_operations_t *list_ops,
                job_base_operations_t *job_ops,
                int NUMBER_OF_JOBS,
                int NUMBER_OF_MACHINES,
                int MAX_SETUP_TIMES,
                weights_t weights,
                std::map<std::pair<std::string, std::string>, double>
                    &transportation_time_table,
                setup_time_parameters_t scheduling_parameters);

/**
 * chromosomeCmp () : The comparison of two chromosomes.
 * If the fitness value of c1 is bigger than c2, return true, or return false.
 * @param _c1
 * @param _c2
 * @return boolean value
 */
int chromosomeCmp(const void *_c1, const void *_c2);

#endif
