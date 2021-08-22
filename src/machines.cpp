#include "include/machines.h"
#include "include/info.h"
#include "include/job_base.h"
#include "include/linked_list.h"
#include "include/machine.h"
#include "include/machine_base.h"
#include "include/parameters.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>


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
    string model_name(machine.model_name.data.text);
    string location(machine.location.data.text);
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

    if (_model_locations.count(model_name) == 0) {
        _model_locations[model_name] = vector<string>();
    }

    if (find(_model_locations[model_name].begin(),
             _model_locations[model_name].end(),
             location) == _model_locations[model_name].end()) {
        _model_locations[model_name].push_back(location);
    }
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
    job_t *job_on_machine;

    for (map<string, machine_t *>::iterator it = _machines.begin();
         it != _machines.end(); ++it) {
        machine_ops->sort_job(&it->second->base, list_ops);
        scheduling(it->second, machine_ops, _weights, _param);
        setJob2Scheduled(it->second);

        // collect the job which is on the machine
        job_on_machine = new job_t();
        *job_on_machine = it->second->current_job;
        _scheduled_jobs.push_back(job_on_machine);

        setLastJobInMachine(
            it->second);  // if the machine is scheduled, it will be set
    }

    // collect scheduled jobs
    list_ele_t *list;
    for (map<string, machine_t *>::iterator it = _machines.begin();
         it != _machines.end(); ++it) {
        list = it->second->base.root;
        while (list) {
            _scheduled_jobs.push_back((job_t *) list->ptr_derived_object);
            list = list->next;
        }
    }
}

bool machinePtrComparison(machine_t *m1, machine_t *m2)
{
    return m1->base.available_time < m2->base.available_time;
}

bool jobPtrComparison(job_t *j1, job_t *j2)
{
    return j1->base.arriv_t < j2->base.arriv_t;
}

void machines_t::addGroupJobs(string recipe, vector<job_t *> jobs)
{
    if (_groups.count(recipe) != 0)
        cerr << "Warning : you add group of jobs twice, recipe is [" << recipe
             << "]" << endl;

    vector<machine_t *> machines;
    for (auto it = _machines.begin(); it != _machines.end(); it++) {
        if (strcmp(it->second->current_job.bdid.data.text, recipe.c_str()) ==
            0) {
            machines.push_back(it->second);
        }
    }

    _groups[recipe] =
        (struct __machine_group_t){.machines = machines,
                                   .unscheduled_jobs = jobs,
                                   .scheduled_jobs = vector<job_t *>()};
}

vector<machine_t *> machines_t::_sortedMachines(vector<machine_t *> &ms)
{
    sort(ms.begin(), ms.end(), machinePtrComparison);
    return ms;
}

vector<job_t *> machines_t::_sortedJobs(std::vector<job_t *> &jobs)
{
    sort(jobs.begin(), jobs.end(), jobPtrComparison);
    return jobs;
}


void machines_t::_scheduleAGroup(struct __machine_group_t *group)
{
    vector<machine_t *> machines = group->machines;
    vector<job_t *> unscheduled_jobs = group->unscheduled_jobs;

    sort(machines.begin(), machines.end(), machinePtrComparison);
    sort(unscheduled_jobs.begin(), unscheduled_jobs.end(), jobPtrComparison);
    bool end = true;
    while (end) {
        int num_scheduled_jobs = 0;
        iter(machines, i)
        {
            string location(machines[i]->location.data.text);
            string model(machines[i]->model_name.data.text);

            iter(unscheduled_jobs, j)
            {
                if (unscheduled_jobs[j]->is_scheduled)
                    continue;

                // check the location is suitable
                string lot_number(unscheduled_jobs[j]->base.job_info.data.text);
                vector<string> locations = _job_can_run_locations[lot_number];

                if (find(locations.begin(), locations.end(), location) !=
                        locations.end() &&
                    (unscheduled_jobs[j]->base.arriv_t -
                         machines[i]->base.available_time <=
                     60)) {
                    unscheduled_jobs[j]->base.ptime =
                        _job_process_times[lot_number][model];
                    staticAddJob(machines[i], unscheduled_jobs[j], machine_ops);
                    unscheduled_jobs[j]->is_scheduled = true;
                    unscheduled_jobs[j]->base.machine_no =
                        machines[i]->base.machine_no;
                    num_scheduled_jobs += 1;
                    break;
                }
            }
        }
        if (num_scheduled_jobs == 0)
            end = false;
    }

    group->unscheduled_jobs.clear();
    group->scheduled_jobs.clear();
    iter(unscheduled_jobs, i)
    {
        if (unscheduled_jobs[i]->is_scheduled) {
            group->scheduled_jobs.push_back(unscheduled_jobs[i]);
        } else {
            group->unscheduled_jobs.push_back(unscheduled_jobs[i]);
        }
    }
}

void machines_t::scheduleGroups()
{
    std::map<std::string, struct __machine_group_t> ngroups;
    for (map<string, struct __machine_group_t>::iterator it = _groups.begin();
         it != _groups.end(); it++) {
        _scheduleAGroup(&it->second);
        _scheduled_jobs += it->second.scheduled_jobs;  // collect scheduled lots
        if (it->second.unscheduled_jobs.size() != 0) {
            ngroups[it->first] = it->second;
            ngroups[it->first].scheduled_jobs.clear();
            ngroups[it->first].machines.clear();
        }
    }
    _groups = ngroups;
    for (map<string, struct __machine_group_t>::iterator it = _groups.begin();
         it != _groups.end(); it++) {
        printf("[%s] : %lu jobs\n", it->first.c_str(),
               it->second.unscheduled_jobs.size());
    }
}