// #include <pthread.h>
// #include <semaphore.h>
// #include <unistd.h>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <map>
#include <string>
#include <thread>
#include "include/machine_base.h"

#define LOG_ERROR

#include "include/algorithm.h"
#include "include/arg_parser.h"
#include "include/csv.h"
#include "include/entity.h"
#include "include/infra.h"
#include "include/lot.h"
#include "include/lots.h"
#include "include/machines.h"
#include "include/population.h"
#include "include/record_gap.h"

using namespace std;

map<string, string> outputJob(job_t job);
map<string, string> outputJobInMachine(machine_t *machine);

void outputJobInMachine(map<string, machine_t *>, csv_t *csv);

lots_t createLots(map<string, string>);
entities_t createEntities(map<string, string>);

typedef struct __thread_data_t {
    map<string, string> arguments;
    int argc;
    const char **argv;
    int fd;
    int id;
} thread_data_t;
void run(thread_data_t *data);

char MESSAGE[] =
    "version 0.0.6\n"
    "Author : NCKU Smart Production Lab";

argument_parser_t *parser = new argument_parser_t();


int main(int argc, const char **argv)
{
    parser->add_args({"", "setup input file", ARG_STRING, NULL},
                     {"-f"s, "--file"s});
    parser->add_args({"", "only preprocess the files", ARG_NONE, NULL},
                     {"-p"s, "--preprocessing"s});
    parser->add_args({"", "list the available arguments\n", ARG_NONE, NULL},
                     {"-h"s, "--help"s});
    parser->add_args(
        {"", "rerun and skip the file preprocessing process", ARG_NONE, NULL},
        {"-r"s, "--rerun"s});

    parser->parse_argument_list(argc, argv);

    if (parser->is_set("-h")) {
        parser->print_arg_description();
    }

    string file_name = parser->get_argument_value("--file");
    if (file_name.length() == 0) {
        printf("%s\n", MESSAGE);
        printf("Please give the --file argument");
        return 0;
    }

    csv_t cfg(file_name, "r", true, true);
    // get the cfg size
    int nthreads = cfg.nrows();
    /*pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * nthreads);*/
    vector<thread> threads;

    thread_data_t **thread_data_array =
        (thread_data_t **) malloc(sizeof(thread_data_t *) * nthreads);


    srand(time(NULL));
    for (unsigned int i = 0; i < cfg.nrows(); ++i) {
        thread_data_array[i] = new thread_data_t();
        *thread_data_array[i] = thread_data_t{.arguments = cfg.getElements(i),
                                              .argc = argc,
                                              .argv = argv,
                                              .fd = 1,
                                              .id = (int) i};
        printf("Thread[%d] starts\n", i);
        threads.push_back(thread(run, thread_data_array[i]));
    }

    for (unsigned int i = 0, size = threads.size(); i < size; ++i) {
        threads[i].join();
        printf("Thread[%d] is finished\n", i);
    }

    return 0;
}

void run(thread_data_t *data)
{
    int id = data->id;
    map<string, string> arguments = data->arguments;

    lots_t lots = createLots(arguments);
    entities_t entities = createEntities(arguments);

    machine_base_operations_t *machine_ops =
        (machine_base_operations_t *) malloc(sizeof(machine_base_operations_t) +
                                             sizeof(setup_time_unit_t) * 7);


    // machine_ops->add_job = machineBaseAddJob;
    // machine_ops->sort_job = machineBaseSortJob;
    // machine_ops->setup_time_functions[0] = {setupTimeCWN,
    // parameters.TIME_CWN}; machine_ops->setup_time_functions[1] =
    // {setupTimeCK, parameters.TIME_CK}; machine_ops->setup_time_functions[2] =
    // {setupTimeEU, parameters.TIME_EU}; machine_ops->setup_time_functions[3] =
    // {setupTimeMC, parameters.TIME_MC}; machine_ops->setup_time_functions[4] =
    // {setupTimeSC, parameters.TIME_SC}; machine_ops->setup_time_functions[5] =
    // {setupTimeCSC, parameters.TIME_CSC}; machine_ops->setup_time_functions[6]
    // = {setupTimeUSC, parameters.TIME_USC};
    // machine_ops->sizeof_setup_time_function_array =
    //     num_of_setup_time_units - 1;  // -1 is for ICSI
    // machine_ops->reset = machineReset;



    if (parser->is_set("-p")) {
        pthread_exit(NULL);
    }

    if (data->fd < 0)
        data->fd = 1;

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
                             stoi(arguments["weight_max_setup_times"]),
                         .WEIGHT_CR = stoi(arguments["weight_cr"])},
             .setup_times_parameters =
                 {
                     .TIME_CWN = stod(arguments["setup_time_cwn"]),
                     .TIME_CK = stod(arguments["setup_time_ck"]),
                     .TIME_EU = stod(arguments["setup_time_eu"]),
                     .TIME_MC = stod(arguments["setup_time_mc"]),
                     .TIME_SC = stod(arguments["setup_time_sc"]),
                     .TIME_CSC = stod(arguments["setup_time_csc"]),
                     .TIME_USC = stod(arguments["setup_time_usc"]),
                     .TIME_ICSI = stod(arguments["setup_time_icsi"]),
                 },
             .scheduling_parameters =
                 {.PEAK_PERIOD = stof(arguments["peak_period"]),
                  .MAX_SETUP_TIMES = stoi(arguments["max_setup_times"]),
                  .MINUTE_THRESHOLD = stoi(arguments["minute_threshold"])}},

    };


    machine_base_operations_initializer_t ops_init(
        {setupTimeCWN, setupTimeCK, setupTimeEU, setupTimeMC, setupTimeSC,
         setupTimeCSC, setupTimeUSC},
        pop.parameters.setup_times_parameters);
    machine_base_operations_t *ops = ops_init.getOps();

    string directory = "output_" + arguments["no"];
    Record_gap rg(ops, directory);

    machines_t *machines = new machines_t(pop.parameters.setup_times_parameters,
                                          pop.parameters.weights);

    machines->setThreshold(
        pop.parameters.scheduling_parameters.MINUTE_THRESHOLD);

    vector<entity_t *> all_entities = entities.allEntities();
    foreach (all_entities, i) {
        machines->addMachine(all_entities[i]->machine());
    }
    machines->setMachineConstraintA(entities.getMachineConstraintA());
    machines->setMachineConstraintR(entities.getMachineConstraintR());

    prescheduling(machines, &lots);
    int stage2_setup_times = stage2Scheduling(
        machines, &lots, pop.parameters.scheduling_parameters.PEAK_PERIOD);
    pop.parameters.scheduling_parameters.MAX_SETUP_TIMES -= stage2_setup_times;
    stage3Scheduling(machines, &lots, &pop, data->fd);
    vector<job_t *> scheduled_jobs = machines->getScheduledJobs();
    csv_t result(directory + "/result.csv", "w");
    foreach (scheduled_jobs, i) {
        result.addData(outputJob(*scheduled_jobs[i]));
        rg.addJob(scheduled_jobs[i]);
    }

    for (int i = 0; i < pop.objects.NUMBER_OF_JOBS; ++i) {
        result.addData(outputJob(*pop.objects.jobs[i]));
        rg.addJob(pop.objects.jobs[i]);
    }

    rg.record_gap_all_machines();
    result.write();
}


lots_t createLots(map<string, string> arguments)
{
    lots_t lots;
    lot_t *lot;
    if (parser->is_set("-r")) {
        string lots_csv_file_path = arguments["lots"];
        csv_t lots_csv(lots_csv_file_path, "r");
        try {
            lots_csv.read();
        } catch (exception &e) {
            cerr << "The pre-created lots.csv is not found, the file path is "
                 << lots_csv_file_path << endl;
            exit(EXIT_FAILURE);
        }
        vector<lot_t *> all_lots;
        for (int i = 0, nrows = lots_csv.nrows(); i < nrows; ++i) {
            lot = new lot_t(lots_csv.getElements(i));
            all_lots.push_back(lot);
        }
        lots.addLots(all_lots);
    } else {
        // printf("Create lots by using configure file : %s\n", argv[1]);
        lots.createLots(arguments);
    }
    lots.setProcessTimeRatio(stod(arguments["process_time_ratio"]));

    return lots;
}

entities_t createEntities(map<string, string> arguments)
{
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
