#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
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
#include "include/progress.h"


using namespace std;

map<string, string> outputJob(job_t job);
map<string, string> outputJobInMachine(machine_t *machine);

void outputJobInMachine(map<string, machine_t *>, csv_t *csv);

lots_t createLots(int argc, const char **, map<string, string>);
entities_t createEntities(int argc, const char **argv, map<string, string>);

void *run(void *);

typedef struct __thread_data_t {
    map<string, string> arguments;
    int argc;
    const char **argv;
    int fd;
} thread_data_t;

sem_t SEM;

int main(int argc, const char **argv)
{
    if (argc < 2) {
        printf("Please specify the path of configuration file\n");
        exit(EXIT_FAILURE);
    }
    sem_init(&SEM, 0, 15);

    csv_t cfg(argv[1], "r", true, true);
    // get the cfg size
    int nthreads = cfg.nrows();
    pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * nthreads);
    thread_data_t **thread_data_array =
        (thread_data_t **) malloc(sizeof(thread_data_t *) * nthreads);


    srand(time(NULL));
    progress_bar_attr_t *pbattr =
        create_progress_bar_attr(nthreads + 1, "127.0.0.1", 8081);
    pthread_t accept_thread, progress_bar_thread;
    pthread_create(&accept_thread, NULL, accept_connection, (void *) pbattr);

    int fds[1024] = {-1};
    for (int i = 0; i < nthreads; ++i) {
        fds[i] = create_client_connection("127.0.0.1", 8081);
    }
    int main_fd = create_client_connection("127.0.0.1", 8081);
    pthread_join(accept_thread, NULL);

    pthread_create(&progress_bar_thread, NULL, run_progress_bar_server,
                   (void *) pbattr);
    // map<string, string> arguments = cfg.getElements(0);
    for (unsigned int i = 0; i < cfg.nrows(); ++i) {
        thread_data_array[i] = new thread_data_t();
        *thread_data_array[i] = thread_data_t{.arguments = cfg.getElements(i),
                                              .argc = argc,
                                              .argv = argv,
                                              .fd = fds[i]};
        pthread_create(threads + i, NULL, run, thread_data_array[i]);
    }

    for (unsigned int i = 0; i < cfg.nrows(); ++i) {
        pthread_join(threads[i], NULL);
    }
    sem_destroy(&SEM);
    send(main_fd, "close", 5, 0);
    pthread_join(progress_bar_thread, NULL);
    delete_attr(&pbattr);


    return 0;
}

void *run(void *_data)
{
    // sem_wait(&SEM);
    thread_data_t *data = (thread_data_t *) _data;
    int argc = data->argc;
    const char **argv = data->argv;
    map<string, string> arguments = data->arguments;

    lots_t lots = createLots(argc, argv, arguments);
    entities_t entities = createEntities(argc, argv, arguments);

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
                             stoi(arguments["weight_max_setup_times"])},
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
                 {.PEAK_PERIOD = stoi(arguments["peak_period"]),
                  .MAX_SETUP_TIMES = stoi(arguments["max_setup_times"]),
                  .MINUTE_THRESHOLD = stoi(arguments["minute_threshold"])}},

    };

    machines_t *machines = new machines_t(pop.parameters.setup_times_parameters,
                                          pop.parameters.weights);

    machines->setThreshold(
        pop.parameters.scheduling_parameters.MINUTE_THRESHOLD);

    vector<entity_t *> all_entities = entities.allEntities();
    foreach (all_entities, i) {
        machines->addMachine(all_entities[i]->machine());
    }

    prescheduling(machines, &lots);
    int stage2_setup_times = stage2Scheduling(
        machines, &lots, pop.parameters.scheduling_parameters.PEAK_PERIOD);
    pop.parameters.scheduling_parameters.MAX_SETUP_TIMES -= stage2_setup_times;
    stage3Scheduling(machines, &lots, &pop, data->fd);
    vector<job_t *> scheduled_jobs = machines->getScheduledJobs();
    string directory = "output_" + arguments["no"];
    csv_t result(directory + "/result.csv", "w");
    foreach (scheduled_jobs, i) {
        result.addData(outputJob(*scheduled_jobs[i]));
    }

    for (int i = 0; i < pop.objects.NUMBER_OF_JOBS; ++i) {
        result.addData(outputJob(*pop.objects.jobs[i]));
    }
    result.write();

    // sem_post(&SEM);
    return NULL;
}


lots_t createLots(int argc, const char **argv, map<string, string> arguments)
{
    lots_t lots;
    lot_t *lot;
    if (argc >= 3) {
        // printf("Create lots by using pre-created lots.csv file : %s\n",
        //        argv[2]);
        csv_t lots_csv(argv[2], "r", true, true);
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

    return lots;
}

entities_t createEntities(int argc,
                          const char **argv,
                          map<string, string> arguments)
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
