#include <ctime>
#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include "include/chromosome_base.h"
#include "include/job_base.h"
#include "include/linked_list.h"
#include "include/machine_base.h"

#include <include/arrival.h>
#include <include/condition_card.h>
#include <include/csv.h>
#include <include/da.h>
#include <include/entity.h>
#include <include/infra.h>
#include <include/lot.h>
#include <include/route.h>
#include <include/population.h>


using namespace std;

round_t createARound(vector<lot_group_t> group, machines_t &machines, ancillary_resources_t & tools, ancillary_resources_t & wires);


void initializeARound(round_t * r);

void initializePopulation(population_t *pop, machines_t & machines, ancillary_resources_t & tools, ancillary_resources_t & wires);


void geneticAlgorithm(population_t * pop);

int main(int argc, const char *argv[])
{
    csv_t lot_csv("lots.csv", "r", true, true);
    lot_csv.setHeaders(map<string, string>({
                    {"route", "route"},
                    {"lot_number", "lot_number"},
                    {"pin_package", "pin_package"},
                    {"bd_id", "recipe"},
                    {"prod_id", "prod_id"},
                    {"part_id", "part_id"}, 
                    {"part_no", "part_no"},
                    {"urgent_code", "urgent_code"},
                    {"qty", "qty"},
                    {"dest_oper", "dest_oper"},
                    {"oper", "dest_oper"},
                    {"hold", "hold"},
                    {"mvin", "mvin"},
                    {"queue_time", "queue_time"},
                    {"fcst_time", "fcst_time"},
                    {"amount_of_tools", "amount_of_tools"},
                    {"amount_of_wires", "amount_of_wires"},
                    {"CAN_RUN_MODELS", "CAN_RUN_MODELS"},
                    {"PROCESS_TIME", "PROCESS_TIME"},
                    {"uphs", "uphs"},
                    {"customer", "customer"}
                }));
    lot_csv.trim(" ");
    vector<lot_t> allots;
    for(int i = 0, nrows = lot_csv.nrows(); i < nrows; ++i){
        allots.push_back(lot_t(lot_csv.getElements(i)));
    } 

    lots_t lots;
    lots.addLots(allots);

        
    // lots.addLots(createLots("WipOutPlanTime_.csv", "product_find_process_id.csv",
    //                "process_find_lot_size_and_entity.csv", "fcst.csv",
    //                "routelist.csv", "newqueue_time.csv",
    //                "BOM List_20210521.csv", "Process find heatblock.csv",
    //                "EMS Heatblock data.csv", "GW Inventory.csv"));


    ancillary_resources_t tools(lots.amountOfTools());
    ancillary_resources_t wires(lots.amountOfWires());

    csv_t machine_csv("machines.csv", "r", true, true);
    machine_csv.trim(" ");
    machine_csv.setHeaders(map<string, string>({{"entity", "ENTITY"},
                                                {"model", "MODEL"},
                                                {"recover_time", "OUTPLAN"}}));

    csv_t location_csv("locations.csv", "r", true, true);
    location_csv.trim(" ");
    location_csv.setHeaders(
        map<string, string>({{"entity", "Entity"}, {"location", "Location"}}));


    char *text = strdup("2020/12/19 10:50");
    entities_t entities(text);
    entities.addMachines(machine_csv, location_csv);
    machines_t machines;
    machines.addMachines(entities.getAllEntity());

    vector<vector<lot_group_t> > round_groups = lots.rounds(entities);
    
    population_t pop = population_t{ 
        .groups = round_groups,
        .current_round_no = 0,
    };

    initializePopulation(&pop, machines, tools, wires);
    
    geneticAlgorithm(&pop);
    
    return 0;
}



double setup_time_CWN(job_base_t * _prev, job_base_t * _next){
    if(_prev){
        job_t * prev = (job_t*)_prev->ptr_derived_object;
        job_t * next = (job_t*)_next->ptr_derived_object;
        if(isSameInfo(prev->part_no, next->part_no))
            return 0.0;
        else
           return 30.0;

    }
    return 0;
}

double setup_time_CK(job_base_t * _prev, job_base_t * _next){
    if(_prev){
        job_t * prev = (job_t*)_prev->ptr_derived_object;
        job_t * next = (job_t*)_next->ptr_derived_object;
        if(prev->part_no.data.text[5] == next->part_no.data.text[5]){
            return 0.0;
        }else{
            return 0;
        }
    }
    return 0; 
}

double setup_time_EU(job_base_t *_prev, job_base_t * _next){
    if(_next){
        job_t * next = (job_t *)_next->ptr_derived_object;
        if(next->urgent_code)
            return 0; // FIXME
        else
            return 0;
    }
    return -1;
}

double setup_time_MC_SC(job_base_t * _prev, job_base_t *_next){
    if(_prev && _next){
        job_t *prev = (job_t*)_prev->ptr_derived_object;
        job_t *next = (job_t *)_next->ptr_derived_object;
        if(prev->pin_package.data.number[0] == next->pin_package.data.number[0]){
            return 84;
        }else{
            return 90;
        }
    }
    return 0;
}

double setup_time_CSC(job_base_t * _prev, job_base_t *_next){
    if(_next){
        job_t *next = (job_t*)_next->ptr_derived_object;
        if(next->part_no.data.text[5] != 'A')
            return 0;
    }
    return 0;
}

double setup_time_USC(job_base_t * _prev, job_base_t *_next){
    if(_next){
        job_t *next = (job_t*)_next->ptr_derived_object;
        if(next->part_no.data.text[5] != 'A' && next->urgent_code == 'Y')
            return 72;
        else
            return 0;

    }
    return -1;
}

double calculateSetupTime(job_t *prev, job_t *next, machine_base_operations_t * ops){
    // first 3 jobcode : CWN->CK->EU
    double time = 0;
    for(unsigned int i = 0; i < ops->sizeof_setup_time_function_array; ++i){
        if(prev)
            time += ops->setup_times[i](&prev->base,&next->base);
        else
            time += ops->setup_times[i](NULL, &next->base);
    }
    return time;
}

void scheduling(machine_t * machine, machine_base_operations_t * ops){
    list_ele_t *it;
    job_t *job;
    job_t *prev_job = NULL;
    it = machine->base.root;
    double arrival_time; 
    double setup_time;
    bool hasICSI = false;
    double start_time = machine->base.avaliable_time;
    while(it){
        job = (job_t *)it->ptr_derived_object;
        arrival_time = job->base.arriv_t;
        setup_time = calculateSetupTime(prev_job, job, ops);
        if(!hasICSI){
            if(strncmp(job->customer.data.text, "ICSI", 4) == 0){
                setup_time += 54;
                hasICSI = true;
            }
        }  

        start_time = (start_time + setup_time) > arrival_time ? start_time + setup_time : arrival_time;
        set_start_time(&job->base, start_time);
        start_time = get_end_time(&job->base);

        prev_job = job;
        it = it->next;
    }
    machine->makespan = start_time;
    machine->tool->time = start_time;
    machine->wire->time = start_time;
}

void geneticAlgorithm(population_t * pop){

    int AMOUNT_OF_JOBS = pop->round.AMOUNT_OF_JOBS;
    job_t * jobs = pop->round.jobs;
    chromosome_base_t * chromosomes = pop->chromosomes; 
    map<unsigned int, machine_t *> machines = pop->round.machines;
    int machine_idx;
    unsigned int machine_no;
    process_time_t tmp;
    
    list_operations_t list_ops = LINKED_LIST_OPS;

    // initialize machine_op
    machine_base_operations_t * machine_ops;
    machine_ops = (machine_base_operations_t *)malloc(sizeof(machine_base_operations_t) + sizeof(setup_time_t)*7);
    machine_ops->add_job = _machine_base_add_job;
    machine_ops->sort_job = _machine_base_sort_job;
    machine_ops->setup_times[0] = setup_time_CWN;
    machine_ops->setup_times[1] = setup_time_CK;
    machine_ops->setup_times[2] = setup_time_EU;
    machine_ops->setup_times[3] = setup_time_MC_SC;
    machine_ops->setup_times[4] = setup_time_CSC;
    machine_ops->setup_times[5] = setup_time_USC;
    machine_ops->sizeof_setup_time_function_array = 6;


    for(int i = 0; i < 100; ++i){ // chromosomes
        printf("decoding chromosomes[%d]\n", i);
        //machine selection
        for(int j = 0; j < AMOUNT_OF_JOBS; ++j){
            set_ms_gene_addr(&jobs[j].base, chromosomes[i].ms_genes + j);
            set_os_gene_addr(&jobs[j].base, chromosomes[i].os_genes + j);
            machine_idx = get_ms_gene(&jobs[j].base) / jobs[j].base.partition;

            tmp = jobs[j].base.process_time[machine_idx];
            jobs[j].base.ptime = tmp.process_time;
            machine_no = tmp.machine_no;
            _machine_base_add_job(&machines[machine_no]->base, &jobs[j].list);
        }

        // sorting;
        for(map<unsigned int, machine_t *>::iterator it = machines.begin(); it != machines.end(); it++){
            _machine_base_sort_job(&it->second->base, &list_ops);
            // machine_base_reset(&it->second->base);
        } 

        //scheduling
        for(map<unsigned int, machine_t *>::iterator it = machines.begin(); it != machines.end(); it++){
            scheduling(it->second, machine_ops); 
            machine_base_reset(&it->second->base);
        }
        printf("end of a generation!\n"); 
    } 
}

void random(double *genes, int size)
{
    for (int i = 0; i < size; ++i) {
        genes[i] = (double) rand() / (double) RAND_MAX;
    }
}

int random_range(int start, int end, int different_num)
{
    if (different_num < 0) {
        return start + rand() % (end - start);
    } else {
        int rnd = start + (rand() % (end - start));
        while (rnd == different_num) {
            rnd = start + (rand() % (end - start));
        }
        return rnd;
    }
}

void initializePopulation(population_t *pop, machines_t & machines, ancillary_resources_t & tools, ancillary_resources_t & wires){
    pop->round = createARound(pop->groups[pop->current_round_no], machines, tools, wires);
    pop->chromosomes = (chromosome_base_t *)malloc(sizeof(chromosome_base_t)*100);
    for(int i = 0; i < 100; ++i){
        pop->chromosomes[i].genes = (double*)malloc(sizeof(double)*pop->round.AMOUNT_OF_JOBS * 2); 
        pop->chromosomes[i].ms_genes = pop->chromosomes[i].genes;
        pop->chromosomes[i].os_genes = pop->chromosomes[i].genes + pop->round.AMOUNT_OF_JOBS;
        random(pop->chromosomes[i].genes, pop->round.AMOUNT_OF_JOBS * 2);
    }

}


round_t prepareResources(vector<lot_group_t> group, machines_t & machines, ancillary_resources_t & tools, ancillary_resources_t &wires){
    int AMOUNT_OF_MACHINES = 0;
    vector<tool_t *> alltools;
    vector<wire_t *> allwires;
    map<string, machine_t *> allmachines = machines.getMachines();
    vector<machine_t *> selected_machines;
    map<unsigned int, machine_t *> machines_map;

   
    vector<tool_t *> ts;
    vector<wire_t *> ws;
    iter(group, i){
        ts = tools.aRound(group[i].tool_name, group[i].machine_amount);
        ws = wires.aRound(group[i].wire_name, group[i].machine_amount);
        iter(group[i].entities, j){
            machine_t * m = allmachines[group[i].entities[j]->entity_name];
            m->tool = ts[j];
            m->wire = ws[j];
            ts[j]->machine_no = convertEntityNameToUInt(group[i].entities[j]->entity_name);
            ws[j]->machine_no = convertEntityNameToUInt(group[i].entities[j]->entity_name);

            // set the recover time max(machine, tool, wire);
            double mx = max(m->base.avaliable_time, m->tool->time);
            mx = max(mx, m->wire->time);

            // TODO: calculate setup time
            m->base.avaliable_time = mx;
            

            machines_map[m->base.machine_no] = m;
            alltools.push_back(ts[j]);
            allwires.push_back(ws[j]);
            AMOUNT_OF_MACHINES += 1;
        }
    }
    
    // machine_t ** machines_ptr = (machine_t**)malloc(sizeof(machine_t*)*AMOUNT_OF_MACHINES);
    // iter(selected_machines, i){
    //     machines_ptr[i] = selected_machines[i];
    // }
   
    // prepare tool
    tool_t ** tools_ar;
    wire_t ** wires_ar;
    tools_ar = (tool_t **)malloc(sizeof(tool_t*)*AMOUNT_OF_MACHINES);
    wires_ar = (wire_t **)malloc(sizeof(wire_t*)*AMOUNT_OF_MACHINES);

    iter(alltools, i){
        tools_ar[i] = alltools[i];
        wires_ar[i] = allwires[i];
    }


    return round_t{
        .AMOUNT_OF_MACHINES = AMOUNT_OF_MACHINES,
        .machines = machines_map,
        .tools = tools_ar,
        .wires = wires_ar
    };
}

round_t prepareJobs(vector<lot_group_t> group){
   int AMOUNT_OF_JOBS = 0;
    vector<lot_t *> lots;
    iter(group, i){
        // iter(group[i].lots, j){
        //     counter++;
        //     lots.push_back(group[i].lots[j]);
        // }
        lots += group[i].lots;
    }

    AMOUNT_OF_JOBS = lots.size();
    job_t *jobs = (job_t *)malloc(sizeof(job_t) * AMOUNT_OF_JOBS);
    process_time_t ** pts = (process_time_t**)malloc(sizeof(process_time_t*)*AMOUNT_OF_JOBS);
    int *size_of_pt = (int*)malloc(sizeof(int)*AMOUNT_OF_JOBS); 
   
    // prepare jobs
    // set process time
    iter(lots, i){
        jobs[i] = lots[i]->job();
        vector<string> can_run_ents = lots[i]->getCanRunEntities();   
        map<string, double> ent_process_time = lots[i]->getEntitiyProcessTime();
        pts[i] = (process_time_t *)malloc(sizeof(process_time_t)* can_run_ents.size());
        size_of_pt[i] = can_run_ents.size();
        iter(can_run_ents, j){
            pts[i][j].machine_no = convertEntityNameToUInt(can_run_ents[j]);
            pts[i][j].process_time = ent_process_time[can_run_ents[j]];
        }
        set_process_time(&jobs[i].base, pts[i], size_of_pt[i]);
        jobs[i].base.ptr_derived_object = &jobs[i];
        jobs[i].list.ptr_derived_object = &jobs[i];
    }
    
    return round_t{
        .AMOUNT_OF_JOBS = AMOUNT_OF_JOBS,
        .jobs = jobs,
        .process_times = pts,
        .size_of_process_times = size_of_pt
    };
}

round_t createARound(vector<lot_group_t> group, machines_t & machines, ancillary_resources_t & tools, ancillary_resources_t & wires){
    round_t round_res = prepareResources(group, machines, tools, wires);
    round_t round_jobs = prepareJobs(group);
    round_res.AMOUNT_OF_JOBS = round_jobs.AMOUNT_OF_JOBS;
    round_res.jobs = round_jobs.jobs;
    round_res.process_times = round_jobs.process_times;
    round_res.size_of_process_times = round_jobs.size_of_process_times;
    return round_res;
}
