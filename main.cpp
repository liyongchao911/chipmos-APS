#include <cstdlib>
#include <ctime>
#include <map>
#include <string>

#define LOG_ERROR

#include "include/algorithm.h"
#include "include/csv.h"
#include "include/entity.h"
#include "include/infra.h"
#include "include/lot.h"
#include "include/lots.h"
#include "include/machines.h"
#include "include/population.h"

using namespace std;

map<string, string> outputJob(job_t job);
map<string, string> outputJobInMachine(machine_t *machine);

void outputJobInMachine(map<string, machine_t *>, csv_t *csv);

lots_t createLots(int argc, const char *argv[]);
entities_t createEntities(int argc, const char *argv[]);

char MESSAGE[] =
    "version 0.0.3\n"
    "Author : NCKU Smart Production Lab";

int main(int argc, const char *argv[])
{
    if (argc < 2) {
        printf("%s\n", MESSAGE);
        printf("Please specify the path of configuration file\n");
        exit(EXIT_FAILURE);
    }

    csv_t cfg(argv[1], "r", true, true);
    map<string, string> arguments = cfg.getElements(0);

    lots_t lots = createLots(argc, argv);
    entities_t entities = createEntities(argc, argv);


    srand(time(NULL));
    population_t pop = population_t{
        .parameters =
            {.AMOUNT_OF_CHROMOSOMES = 100,
             .AMOUNT_OF_R_CHROMOSOMES = 200,
             .EVOLUTION_RATE = 0.8,
             .SELECTION_RATE = 0.2,
             .GENERATIONS = stoi(arguments["times"]),
             .MAX_SETUP_TIMES = stoi(arguments["max_setup_times"]),
             .weights = {.WEIGHT_SETUP_TIMES =
                             stoi(arguments["weight_setup_times"]),
                         .WEIGHT_TOTAL_COMPLETION_TIME =
                             stoi(arguments["weight_total_completion_time"]),
                         .WEIGHT_MAX_SETUP_TIMES =
                             stoi(arguments["weight_max_setup_times"])},
             .scheduling_parameters =
                 {
                     .TIME_CWN = stod(arguments["setup_time_cwn"]),
                     .TIME_CK = stod(arguments["setup_time_ck"]),
                     .TIME_EU = stod(arguments["setup_time_eu"]),
                     .TIME_MC = stod(arguments["setup_time_mc"]),
                     .TIME_SC = stod(arguments["setup_time_sc"]),
                     .TIME_CSC = stod(arguments["setup_time_csc"]),
                     .TIME_USC = stod(arguments["setup_time_usc"]),
                     .TIME_ICSI = stod(arguments["setup_time_icsi"]),
                 }},
    };

    machines_t *machines = new machines_t(pop.parameters.scheduling_parameters,
                                          pop.parameters.weights);

    machines->setThreshold(stoi(arguments["minute_threshold"]));

    vector<entity_t *> all_entities = entities.allEntities();
    foreach (all_entities, i) {
        machines->addMachine(all_entities[i]->machine());
    }
    double peak_period = stof(arguments["peak_period"]);
    prescheduling(machines, &lots);
    stage2Scheduling(machines, &lots, peak_period);
    stage3Scheduling(machines, &lots, &pop);
    vector<job_t *> scheduled_jobs = machines->getScheduledJobs();
    csv_t result("./output/result.csv", "w");
    foreach (scheduled_jobs, i) {
        result.addData(outputJob(*scheduled_jobs[i]));
    }

    for (int i = 0; i < pop.objects.NUMBER_OF_JOBS; ++i) {
        result.addData(outputJob(*pop.objects.jobs[i]));
    }
    result.write();
    return 0;
}



lots_t createLots(int argc, const char *argv[])
{
    csv_t cfg(argv[1], "r", true, true);
    map<string, string> arguments = cfg.getElements(0);

    lots_t lots;
    lot_t *lot;
    if (argc >= 3) {
        printf("Create lots by using pre-created lots.csv file : %s\n",
               argv[2]);
        csv_t lots_csv(argv[2], "r", true, true);
        vector<lot_t *> all_lots;
        for (int i = 0, nrows = lots_csv.nrows(); i < nrows; ++i) {
            lot = new lot_t(lots_csv.getElements(i));
            all_lots.push_back(lot);
        }
        lots.addLots(all_lots);
    } else {
        printf("Create lots by using configure file : %s\n", argv[1]);
        lots.createLots(arguments);
    }

    return lots;
}

entities_t createEntities(int argc, const char *argv[])
{
    csv_t cfg(argv[1], "r", true, true);
    map<string, string> arguments = cfg.getElements(0);

    csv_t machine_csv(arguments["machines"], "r", true, true);
    machine_csv.trim(" ");
    machine_csv.setHeaders(map<string, string>({{"entity", "ENTITY"},
                                                {"model", "MODEL"},
                                                {"recover_time", "OUTPLAN"},
                                                {"in_time", "IN TIME"},
                                                {"prod_id", "PRODUCT"},
                                                {"pin_package", "PIN_PKG"},
                                                {"lot_number", "LOT#"},
                                                {"customer", "CUST"},
                                                {"bd_id", "BOND ID"},
                                                {"oper", "OPER"},
                                                {"qty", "WIP"}}));

    csv_t location_csv(arguments["locations"], "r", true, true);
    location_csv.trim(" ");
    location_csv.setHeaders(
        map<string, string>({{"entity", "Entity"}, {"location", "Location"}}));


    entities_t entities(arguments);
    entities.addMachines(machine_csv, location_csv);
    return entities;
}

map<string, string> outputJob(job_t job)
{
    return map<string, string>({{"lot_number", job.base.job_info.data.text},
                                {"bd_id", job.bdid.data.text},
                                {"part_no", job.part_no.data.text},
                                {"part_id", job.part_id.data.text},
                                {"cust", job.customer.data.text},
                                {"pin_pkg", job.pin_package.data.text},
                                {"prod_id", job.prod_id.data.text},
                                {"qty", to_string(job.base.qty)},
                                {"entity", job.base.machine_no.data.text},
                                {"start_time", to_string(job.base.start_time)},
                                {"end_time", to_string(job.base.end_time)},
                                {"oper", to_string(job.oper)},
                                {"arrival_time", to_string(job.base.arriv_t)},
                                {"process_time", to_string(job.base.ptime)}});
}

map<string, string> outputJobInMachine(machine_t *machine)
{
    return outputJob(machine->current_job);
}

void outputJobInMachine(map<string, machine_t *> machines, csv_t *csv)
{
    for (map<string, machine_t *>::iterator it = machines.begin();
         it != machines.end(); it++) {
        if (it->second->current_job.prod_id.text_size != 0)
            csv->addData(outputJob(it->second->current_job));
    }
}
