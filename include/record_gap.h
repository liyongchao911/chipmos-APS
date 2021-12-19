#ifndef __RECORD_GAP_H__
#define __RECORD_GAP_H__

#include <map>
#include <string>
#include <vector>

#include "include/csv.h"
#include "include/info.h"
#include "include/job.h"
#include "include/machine.h"

#define __SETUP(name)          \
    {                          \
#name, setupTime##name \
    }

class Record_gap
{
private:
    std::map<std::string, std::vector<job_t *>> _jobs;
    machine_base_operations_t *ops;

    std::string directory_name;
    csv_t csv_file;

    std::string str;
    typedef struct name_function_pair_t {
        char name[4];
        setup_time_t function;
    } name_function_pair_t;

    double calculateSetupTime(job_t *prev,
                              job_t *next,
                              machine_base_operations_t *ops);
    void record_gap_single_machine(std::vector<job_t *> jobs);
    name_function_pair_t nfp[7];

public:
    double const threshold;
    Record_gap(machine_base_operations_t *ops,
               double th = 0.00000001,
               std::string dir_name = /*"output_0-0"*/ "");
    void addJob(job_t *job);
    void record_gap_all_machines();
};
#endif
