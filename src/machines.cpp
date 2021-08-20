#include "include/machines.h"
#include "include/job_base.h"
#include "include/linked_list.h"
#include "include/machine_base.h"
#include "include/parameters.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>

using namespace std;

machines_t::machines_t()
{
    scheduling_parameters_t param;
    weights_t weights;
    memset(&param, 0, sizeof(param));
    _param = param;

    memset(&weights, 0, sizeof(weights));
    _weights = weights;

    _init(_param);
}

machines_t::machines_t(scheduling_parameters_t param, weights_t weights)
{
    _init(param);
    _weights = weights;
}

machines_t::~machines_t()
{
    for (map<string, machine_t *>::iterator it = _machines.begin();
         it != _machines.end(); it++) {
        delete it->second;
    }

    free(list_ops);
    free(machine_ops);
    free(job_ops);
}

machines_t::machines_t(machines_t &other)
{
    throw invalid_argument("copy constructor hasn't been implemented");
}

void machines_t::_init(scheduling_parameters_t parameters)
{
    list_ops = (list_operations_t *) malloc(sizeof(list_operations_t));
    *list_ops = LINKED_LIST_OPS;

    job_ops = (job_base_operations_t *) malloc(sizeof(job_base_operations_t));
    *job_ops = JOB_BASE_OPS;

    size_t num_of_setup_time_units =
        sizeof(scheduling_parameters_t) / sizeof(double);

    machine_ops = (machine_base_operations_t *) malloc(
        sizeof(machine_base_operations_t) +
        sizeof(setup_time_unit_t) * num_of_setup_time_units);

    machine_ops->add_job = machineBaseAddJob;
    machine_ops->sort_job = machineBaseSortJob;
    machine_ops->setup_time_functions[0] = {setupTimeCWN, parameters.TIME_CWN};
    machine_ops->setup_time_functions[1] = {setupTimeCK, parameters.TIME_CK};
    machine_ops->setup_time_functions[2] = {setupTimeEU, parameters.TIME_EU};
    machine_ops->setup_time_functions[3] = {setupTimeMC, parameters.TIME_MC};
    machine_ops->setup_time_functions[4] = {setupTimeSC, parameters.TIME_SC};
    machine_ops->setup_time_functions[5] = {setupTimeCSC, parameters.TIME_CSC};
    machine_ops->setup_time_functions[6] = {setupTimeUSC, parameters.TIME_USC};
    machine_ops->sizeof_setup_time_function_array =
        num_of_setup_time_units - 1;  // -1 is for ICSI
    machine_ops->reset = machineReset;
}


void machines_t::addMachine(machine_t machine)
{
    // check if machine exist
    string machine_name(machine.base.machine_no.data.text);
    if (_machines.count(machine_name) == 1) {
        throw std::invalid_argument("[" + machine_name + "] is added twice");
    }


    machine_t *machine_ptr = nullptr;
    machine_ptr = new machine_t;
    if (machine_ptr == nullptr) {
        perror("Failed to new a machine instance");
        exit(EXIT_FAILURE);
    }
    *machine_ptr = machine;

    machine_ptr->current_job.base.ptr_derived_object =
        &machine_ptr->current_job;
    machine_ptr->current_job.list.ptr_derived_object =
        &machine_ptr->current_job;

    // add into container
    _machines[machine_name] = machine_ptr;


    string part_id(machine_ptr->current_job.part_id.data.text),
        part_no(machine_ptr->current_job.part_no.data.text);

    _tool_machines[part_no].push_back(machine_ptr);
    _wire_machines[part_id].push_back(machine_ptr);
    _tool_wire_machines[part_no + "_" + part_id].push_back(machine_ptr);
}

void machines_t::addPrescheduledJob(job_t *job)
{
    job->base.ptr_derived_object = job;
    job->list.ptr_derived_object = job;
    string machine_no(job->base.machine_no.data.text);
    machine_ops->add_job(&_machines.at(machine_no)->base, &job->list);
}

void machines_t::prescheduleJobs()
{
    for (map<string, machine_t *>::iterator it = _machines.begin();
         it != _machines.end(); ++it) {
        machine_ops->sort_job(&it->second->base, list_ops);
    }

    for (map<string, machine_t *>::iterator it = _machines.begin();
         it != _machines.end(); ++it) {
        scheduling(it->second, machine_ops, _weights, _param);
    }
}
