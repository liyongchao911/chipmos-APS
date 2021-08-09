#include <ctime>
#include <map>
#include <string>

#include "include/csv.h"
#include "include/da.h"
#include "include/entity.h"
#include "include/infra.h"
#include "include/lot.h"
#include "include/population.h"
#include "include/lots.h"
#include "include/algorithm.h"

using namespace std;

void output(population_t * pop, csv_t * csv);

void outputJobInMachine(map<string, machine_t*> , csv_t *csv);

int main(int argc, const char *argv[])
{
    if(argc < 2){
        printf("Please specify the path of configuration file\n");
        exit(1);
    }

    csv_t cfg(argv[1], "r", true, true);
    map<string, string> arguments = cfg.getElements(0);

    lots_t lots;
    lots.createLots(arguments);

    ancillary_resources_t tools(lots.amountOfTools());
    ancillary_resources_t wires(lots.amountOfWires());

    csv_t machine_csv(arguments["machines"], "r", true, true);
    machine_csv.trim(" ");
    machine_csv.setHeaders(map<string, string>({{"entity", "ENTITY"},
                                                {"model", "MODEL"},
                                                {"recover_time", "OUTPLAN"},
                                                {"prod_id", "PRODUCT"},
                                                {"pin_pkg", "PIN_PKG"},
                                                {"lot_number", "LOT#"},
                                                {"customer", "CUST"},
                                                {"bd_id", "BOND ID"},
                                                {"oper", "OPER"}}));

    csv_t location_csv(arguments["locations"], "r", true, true);
    location_csv.trim(" ");
    location_csv.setHeaders(
        map<string, string>({{"entity", "Entity"}, {"location", "Location"}}));


    entities_t entities(arguments);
    entities.addMachines(machine_csv, location_csv);
    machines_t machines;
    machines.addMachines(entities.getAllEntity());
    
    
    // srand(time(NULL));
    population_t pop = population_t{
        .parameters = {.AMOUNT_OF_CHROMOSOMES = 100,
                       .AMOUNT_OF_R_CHROMOSOMES = 200,
                       .EVOLUTION_RATE = 0.8,
                       .SELECTION_RATE = 0.2,
                       .GENERATIONS = stoi(arguments["times"]),
                       .MAX_SETUP_TIMES = stoi(arguments["max_setup_times"])
        },
    };

    csv_t result("output/result.csv", "w");
    outputJobInMachine(machines.getMachines(), &result);
    initializeOperations(&pop);
    
    int i = 0;
    while(lots.toolWireLotsHasLots()){
        printf("i = %d\n", i++);
        pop.groups = lots.round(entities);
        // if(pop.groups.size() == 0){
        //     continue;
        // }
        initializePopulation(&pop, machines, tools, wires);
        geneticAlgorithm(&pop);
        // optimization(&pop);
        output(&pop, &result);
        machineWriteBackToEntity(&pop);
        freeJobs(&pop.round);
        freeResources(&pop.round);
        freeChromosomes(&pop);
    }
    result.write();

    return 0;
}

map<string, string> outputJob(job_t job){
    return map<string, string>({
        {"lot_number" , job.base.job_info.data.text },
        {"bd_id", job.bdid.data.text },
        {"part_no", job.part_no.data.text },
        {"part_id" , job.part_id.data.text },
        {"cust", job.customer.data.text },
        {"pin_pkg", job.pin_package.data.text },
        {"prod_id", job.prod_id.data.text },
        {"qty", to_string(job.base.qty) },
        {"entity", convertUIntToEntityName(job.base.machine_no) },
        {"start_time", to_string(job.base.start_time) },
        {"end_time", to_string(job.base.end_time) },
        {"oper", to_string(job.oper) },
        {"process_time", to_string(job.base.ptime)}
    });
}

void outputJobInMachine(map<string, machine_t *> machines, csv_t *csv){
    for(map<string, machine_t *>::iterator it = machines.begin(); it != machines.end(); it++){
        if(it->second->current_job.prod_id.text_size != 0)
            csv->addData(outputJob(it->second->current_job)); 
    }
}

void output(population_t * pop, csv_t *csv){
    int AMOUNT_OF_JOBS = pop->round.AMOUNT_OF_JOBS;
    job_t *jobs = pop->round.jobs;
    for(int i = 0; i < AMOUNT_OF_JOBS; ++i){
        csv->addData(outputJob(jobs[i]));
    }
}
