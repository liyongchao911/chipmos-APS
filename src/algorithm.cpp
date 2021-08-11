//
// Created by eugene on 2021/7/5.
//

#include <cstdlib>

#include "include/algorithm.h"
#include "include/chromosome.h"
#include "include/chromosome_base.h"
#include "include/infra.h"
#include "include/machine.h"
#include "include/machine_base.h"
#include "include/population.h"

using namespace std;

void initializeOperations(population_t *pop)
{
    machine_base_operations_t *machine_ops;
    machine_ops = (machine_base_operations_t *) malloc(
        sizeof(machine_base_operations_t) + sizeof(setup_time_unit_t) * 7);
    machine_ops->add_job = machineBaseAddJob;
    machine_ops->sort_job = machineBaseSortJob;
    machine_ops->setup_time_functions[0] = {
        setupTimeCWN, pop->parameters.scheduling_parameters.TIME_CWN};
    machine_ops->setup_time_functions[1] = {
        setupTimeCK, pop->parameters.scheduling_parameters.TIME_CK};
    machine_ops->setup_time_functions[2] = {
        setupTimeEU, pop->parameters.scheduling_parameters.TIME_EU};
    machine_ops->setup_time_functions[3] = {
        setupTimeMC, pop->parameters.scheduling_parameters.TIME_MC};
    machine_ops->setup_time_functions[4] = {
        setupTimeSC, pop->parameters.scheduling_parameters.TIME_SC};
    machine_ops->setup_time_functions[5] = {
        setupTimeCSC, pop->parameters.scheduling_parameters.TIME_CSC};
    machine_ops->setup_time_functions[6] = {
        setupTimeUSC, pop->parameters.scheduling_parameters.TIME_USC};
    machine_ops->sizeof_setup_time_function_array = 7;
    machine_ops->reset = machineReset;

    list_operations_t *list_ops;
    list_ops = (list_operations_t *) malloc(sizeof(list_operations_t));
    *list_ops = LINKED_LIST_OPS;

    job_base_operations_t *job_ops;
    job_ops = (job_base_operations_t *) malloc(sizeof(job_base_operations_t));
    *job_ops = JOB_BASE_OPS;

    pop->operations.machine_ops = machine_ops;
    pop->operations.list_ops = list_ops;
    pop->operations.job_ops = job_ops;
}

void prepareChromosomes(chromosome_base_t **_chromosomes,
                        int AMOUNT_OF_JOBS,
                        int AMOUNT_OF_R_CHROMOSOMES)
{
    chromosome_base_t *chromosomes = (chromosome_base_t *) malloc(
        sizeof(chromosome_base_t) * AMOUNT_OF_R_CHROMOSOMES);
    for (int i = 0; i < AMOUNT_OF_R_CHROMOSOMES; ++i) {
        chromosomes[i].chromosome_no = i;
        chromosomes[i].gene_size = AMOUNT_OF_JOBS << 1;
        chromosomes[i].genes =
            (double *) malloc(sizeof(double) * chromosomes[i].gene_size);
        chromosomes[i].ms_genes = chromosomes[i].genes;
        chromosomes[i].os_genes = chromosomes[i].genes + AMOUNT_OF_JOBS;
        random(chromosomes[i].genes, chromosomes[i].gene_size);
    }
    *_chromosomes = chromosomes;
}

void initializePopulation(population_t *pop,
                          machines_t &machines,
                          ancillary_resources_t &tools,
                          ancillary_resources_t &wires)
{
    pop->round = createARound(pop->groups, machines, tools, wires);
    prepareChromosomes(&pop->chromosomes, pop->round.AMOUNT_OF_JOBS,
                       pop->parameters.AMOUNT_OF_R_CHROMOSOMES);
    prepareChromosomes(&pop->tmp_chromosomes, pop->round.AMOUNT_OF_JOBS,
                       pop->parameters.AMOUNT_OF_R_CHROMOSOMES);
}


round_t prepareResources(vector<lot_group_t> group,
                         machines_t &machines,
                         ancillary_resources_t &tools,
                         ancillary_resources_t &wires)
{
    int AMOUNT_OF_MACHINES = 0;
    vector<tool_t *> alltools;
    vector<wire_t *> allwires;
    map<string, machine_t *> allmachines = machines.getMachines();
    vector<machine_t *> selected_machines;
    map<unsigned int, machine_t *> machines_map;


    vector<tool_t *> ts;
    vector<wire_t *> ws;
    iter(group, i)
    {
        ts = tools.aRound(group[i].tool_name, group[i].machine_amount);
        ws = wires.aRound(group[i].wire_name, group[i].machine_amount);
        iter(group[i].entities, j)
        {
            machine_t *m = allmachines[group[i].entities[j]->entity_name];
            m->base.ptr_derived_object = m;
            m->tool = ts[j];
            m->wire = ws[j];
            strcpy(ts[j]->name.data.text, group[i].tool_name.c_str());
            strcpy(ws[j]->name.data.text, group[i].wire_name.c_str());
            ts[j]->machine_no =
                convertEntityNameToUInt(group[i].entities[j]->entity_name);
            ws[j]->machine_no =
                convertEntityNameToUInt(group[i].entities[j]->entity_name);

            // set the recover time max(machine, tool, wire);
            double mx = max(m->base.available_time, m->tool->time);
            mx = max(mx, m->wire->time);
            // TODO: calculate setup time
            m->base.available_time = mx;

            machines_map[m->base.machine_no] = m;
            alltools.push_back(ts[j]);
            allwires.push_back(ws[j]);
            AMOUNT_OF_MACHINES += 1;
        }
    }

    // prepare tool
    tool_t **tools_ar;
    wire_t **wires_ar;
    tools_ar = (tool_t **) malloc(sizeof(tool_t *) * AMOUNT_OF_MACHINES);
    wires_ar = (wire_t **) malloc(sizeof(wire_t *) * AMOUNT_OF_MACHINES);

    iter(alltools, i)
    {
        tools_ar[i] = alltools[i];
        wires_ar[i] = allwires[i];
    }


    return round_t{.AMOUNT_OF_MACHINES = AMOUNT_OF_MACHINES,
                   .machines = machines_map,
                   .tools = tools_ar,
                   .wires = wires_ar};
}

round_t prepareJobs(vector<lot_group_t> group)
{
    int AMOUNT_OF_JOBS = 0;
    vector<lot_t *> lots;
    iter(group, i) { lots += group[i].lots; }

    AMOUNT_OF_JOBS = lots.size();
    job_t *jobs = (job_t *) malloc(sizeof(job_t) * AMOUNT_OF_JOBS);
    process_time_t **pts =
        (process_time_t **) malloc(sizeof(process_time_t *) * AMOUNT_OF_JOBS);
    int *size_of_pt = (int *) malloc(sizeof(int) * AMOUNT_OF_JOBS);

    // prepare jobs
    // set process time
    iter(lots, i)
    {
        jobs[i] = lots[i]->job();
        vector<string> can_run_ents = lots[i]->getCanRunEntities();
        map<string, double> ent_process_time = lots[i]->getEntityProcessTime();
        pts[i] = (process_time_t *) malloc(sizeof(process_time_t) *
                                           can_run_ents.size());
        size_of_pt[i] = can_run_ents.size();
        iter(can_run_ents, j)
        {
            pts[i][j].machine_no = convertEntityNameToUInt(can_run_ents[j]);
            pts[i][j].process_time = ent_process_time[can_run_ents[j]];
        }
        set_process_time(&jobs[i].base, pts[i], size_of_pt[i]);
        jobs[i].base.ptr_derived_object = &jobs[i];
        jobs[i].list.ptr_derived_object = &jobs[i];
    }

    return round_t{.AMOUNT_OF_JOBS = AMOUNT_OF_JOBS,
                   .jobs = jobs,
                   .process_times = pts,
                   .size_of_process_times = size_of_pt};
}

void freeJobs(round_t *round)
{
    free(round->jobs);
    for (int i = 0; i < round->AMOUNT_OF_JOBS; ++i) {
        free(round->process_times[i]);
    }
    free(round->process_times);
    free(round->size_of_process_times);

    round->jobs = nullptr;
    round->process_times = nullptr;
    round->size_of_process_times = nullptr;
}

void freeResources(round_t *round)
{
    free(round->tools);
    free(round->wires);
    round->tools = nullptr;
    round->wires = nullptr;
}

void freeChromosomes(population_t *pop)
{
    for (int i = 0; i < pop->parameters.AMOUNT_OF_R_CHROMOSOMES; ++i) {
        free(pop->chromosomes[i].genes);
    }
    free(pop->chromosomes);
    pop->chromosomes = nullptr;
}

round_t createARound(vector<lot_group_t> group,
                     machines_t &machines,
                     ancillary_resources_t &tools,
                     ancillary_resources_t &wires)
{
    round_t round_res = prepareResources(group, machines, tools, wires);
    round_t round_jobs = prepareJobs(group);
    round_res.AMOUNT_OF_JOBS = round_jobs.AMOUNT_OF_JOBS;
    round_res.jobs = round_jobs.jobs;
    round_res.process_times = round_jobs.process_times;
    round_res.size_of_process_times = round_jobs.size_of_process_times;
    return round_res;
}

chromosome_base_t searchChromosome(double rnd,
                                   vector<chromosome_linker_t> linkers)
{
    iter(linkers, i)
    {
        if (linkers[i].value > rnd)
            return linkers[i].chromosome;
    }
    return linkers[0].chromosome;
}

void chromosomeSelection(chromosome_base_t *chromosomes,
                         chromosome_base_t *tmp_chromosomes,
                         double elites_rate,
                         int AMOUNT_OF_CHROMOSOMES,
                         int AMOUNT_OF_R_CHROMOSOMES)
{
    double sum0 = 0, sum1 = 0;
    double accumulate = 0;
    double rnd = 0;
    int elites_amount = AMOUNT_OF_CHROMOSOMES * elites_rate;
    int random_amount = AMOUNT_OF_CHROMOSOMES - elites_amount;
    vector<chromosome_linker_t> linkers;

    for (int i = elites_amount; i < AMOUNT_OF_R_CHROMOSOMES; ++i)
        sum0 += chromosomes[i].fitnessValue;

    for (int i = elites_amount; i < AMOUNT_OF_R_CHROMOSOMES; ++i) {
        chromosomes[i].fitnessValue = sum0 / chromosomes[i].fitnessValue;
        sum1 += chromosomes[i].fitnessValue;
    }

    for (int i = elites_amount, j = 0; i < AMOUNT_OF_R_CHROMOSOMES; ++i, ++j) {
        copyChromosome(tmp_chromosomes[j], chromosomes[i]);
        linkers.push_back(chromosome_linker_t{
            .chromosome = tmp_chromosomes[j],
            .value = accumulate += chromosomes[i].fitnessValue / sum1});
    }

    for (int i = 0, j = elites_amount; i < random_amount; ++i, ++j) {
        rnd = randomDouble();
        chromosome_base_t selected_chromosome = searchChromosome(rnd, linkers);
        copyChromosome(chromosomes[j], selected_chromosome);
    }
}

void geneticAlgorithm(population_t *pop)
{
    int AMOUNT_OF_JOBS = pop->round.AMOUNT_OF_JOBS;
    int MAX_SETUP_TIMES = pop->parameters.MAX_SETUP_TIMES;
    job_t *jobs = pop->round.jobs;
    chromosome_base_t *chromosomes = pop->chromosomes;
    chromosome_base_t *tmp_chromosomes = pop->tmp_chromosomes;
    map<unsigned int, machine_t *> machines = pop->round.machines;

    // ops
    machine_base_operations_t *machine_ops = pop->operations.machine_ops;
    list_operations_t *list_ops = pop->operations.list_ops;
    job_base_operations_t *job_ops = pop->operations.job_ops;
    // initialize machine_op
    int k;
    for (k = 0; k < pop->parameters.GENERATIONS; ++k) {
        for (int i = 0; i < pop->parameters.AMOUNT_OF_R_CHROMOSOMES;
             ++i) {  // for all chromosomes
            chromosomes[i].fitnessValue = decoding(
                chromosomes[i], jobs, machines, machine_ops, list_ops, job_ops,
                AMOUNT_OF_JOBS, MAX_SETUP_TIMES, pop->parameters.weights,
                pop->parameters.scheduling_parameters);
        }
        // sort the chromosomes
        qsort(chromosomes, pop->parameters.AMOUNT_OF_R_CHROMOSOMES,
              sizeof(chromosomes[0]), chromosomeCmp);
        printf("%d,%.3f\n", k, chromosomes[0].fitnessValue);

        // statistic
        chromosomeSelection(chromosomes, tmp_chromosomes,
                            pop->parameters.SELECTION_RATE,
                            pop->parameters.AMOUNT_OF_CHROMOSOMES,
                            pop->parameters.AMOUNT_OF_CHROMOSOMES);

        // evolution
        // crossover
        int crossover_amount = pop->parameters.AMOUNT_OF_CHROMOSOMES *
                               pop->parameters.EVOLUTION_RATE;
        for (int l = pop->parameters.AMOUNT_OF_CHROMOSOMES;
             l < crossover_amount + pop->parameters.AMOUNT_OF_CHROMOSOMES;
             l += 2) {
            int rnd1 =
                randomRange(0, pop->parameters.AMOUNT_OF_CHROMOSOMES, -1);
            int rnd2 =
                randomRange(0, pop->parameters.AMOUNT_OF_CHROMOSOMES, rnd1);
            crossover(chromosomes[rnd1], chromosomes[rnd2], chromosomes[l],
                      chromosomes[l + 1]);
        }
        // mutation
        for (int l = pop->parameters.AMOUNT_OF_CHROMOSOMES + crossover_amount;
             l < pop->parameters.AMOUNT_OF_R_CHROMOSOMES; ++l) {
            int rnd = randomRange(0, pop->parameters.AMOUNT_OF_CHROMOSOMES, -1);
            mutation(chromosomes[rnd], chromosomes[l]);
        }
    }

    decoding(chromosomes[0], jobs, machines, machine_ops, list_ops, job_ops,
             AMOUNT_OF_JOBS, MAX_SETUP_TIMES, pop->parameters.weights,
             pop->parameters.scheduling_parameters);

    // update machines' avaliable time and set the last job
    for (map<unsigned int, machine_t *>::iterator it = machines.begin();
         it != machines.end(); ++it) {
        it->second->base.available_time = it->second->makespan;
        setLastJobInMachine(it->second);
    }
}

void machineWriteBackToEntity(population_t *pop)
{
    map<unsigned int, machine_t *> machines = pop->round.machines;
    for (map<unsigned int, machine_t *>::iterator it = machines.begin();
         it != machines.end(); it++) {
        entity_t *ent = (entity_t *) it->second->ptr_derived_object;
        ent->recover_time = it->second->makespan;
        ent->job = it->second->current_job;
    }
}
