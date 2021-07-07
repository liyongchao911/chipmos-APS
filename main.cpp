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
                                                {"lot_number", "LOT#"}}));

    csv_t location_csv(arguments["locations"], "r", true, true);
    location_csv.trim(" ");
    location_csv.setHeaders(
        map<string, string>({{"entity", "Entity"}, {"location", "Location"}}));


    entities_t entities(arguments["std_time"]);
    entities.addMachines(machine_csv, location_csv);
    machines_t machines;
    machines.addMachines(entities.getAllEntity());
    
    srand(time(NULL));
    vector<vector<lot_group_t> > round_groups = lots.rounds(entities);
    population_t pop = population_t{
        .parameters = {.AMOUNT_OF_CHROMOSOMES = 100,
                       .AMOUNT_OF_R_CHROMOSOMES = 200,
                       .EVOLUTION_RATE = 0.8,
                       .SELECTION_RATE = 0.2,
                       .GENERATIONS = 2000},
        .groups = round_groups,
        .current_round_no = 0
    };

    csv_t result("result.csv", "w");
    initializeOperations(&pop);
    iter(pop.groups, i){
        initializePopulation(&pop, machines, tools, wires, i);
        geneticAlgorithm(&pop);
        output(&pop, &result);
        freeJobs(&pop.round);
        freeResources(&pop.round);
        freeChromosomes(&pop);
    }
    result.write();

    return 0;
}

void output(population_t * pop, csv_t *csv){
    int AMOUNT_OF_JOBS = pop->round.AMOUNT_OF_JOBS;
    job_t *jobs = pop->round.jobs;
    for(int i = 0; i < AMOUNT_OF_JOBS; ++i){
        csv->addData(map<string, string>({
                        {"lot_number" , jobs[i].base.job_info.data.text },
                        {"bd_id", jobs[i].bdid.data.text },
                        {"part_no", jobs[i].part_no.data.text },
                        {"part_id" , jobs[i].part_id.data.text },
                        {"cust", jobs[i].customer.data.text },
                        {"pin_pkg", jobs[i].pin_package.data.text },
                        {"prod_id", jobs[i].prod_id.data.text },
                        {"qty", to_string(jobs[i].base.qty) },
                        {"entity", convertUIntToEntityName(jobs[i].base.machine_no) },
                        {"start_time", to_string(jobs[i].base.start_time) },
                        {"end_time", to_string(jobs[i].base.end_time) }
                    }));
    }
}
