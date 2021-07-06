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
                       .GENERATIONS = 50},
        .groups = round_groups,
        .current_round_no = 0
    };

    // srand(time(nullptr));
    initializeOperations(&pop);
    iter(pop.groups, i){
        initializePopulation(&pop, machines, tools, wires, i);
        geneticAlgorithm(&pop);
        freeJobs(&pop.round);
        freeResources(&pop.round);
        freeChromosomes(&pop);
    }

    return 0;
}
