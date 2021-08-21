#ifndef __MACHINES_H__
#define __MACHINES_H__

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "include/job_base.h"
#include "include/machine.h"
#include "include/machine_base.h"
#include "include/parameters.h"

struct __machine_group_t {
    std::vector<machine_t *> machines;
    std::vector<job_t *> unscheduled_jobs;
    std::vector<job_t *> scheduled_jobs;
};

class machines_t
{
protected:
    // entity_name -> machine_t *
    std::map<std::string, machine_t *> _machines;

    // tool_wire_name -> machine_t *
    std::map<std::string, std::vector<machine_t *> > _tool_wire_machines;
    std::map<std::string, std::vector<machine_t *> > _tool_machines;
    std::map<std::string, std::vector<machine_t *> > _wire_machines;

    // model->locations
    std::map<std::string, std::vector<std::string> > _model_locations;

    list_operations_t *list_ops;
    machine_base_operations_t *machine_ops;
    job_base_operations_t *job_ops;

    scheduling_parameters_t _param;
    weights_t _weights;

    std::map<std::string, std::vector<std::string> > _job_can_run_locations;
    std::map<std::string, std::map<std::string, double> > _job_process_times;

    std::map<std::string, struct __machine_group_t> _groups;


    void _init(scheduling_parameters_t param);

    std::vector<machine_t *> _sortedMachines(std::vector<machine_t *> &ms);
    std::vector<job_t *> _sortedJobs(std::vector<job_t *> &jobs);

    void _scheduleAGroup(struct __machine_group_t *group);

public:
    machines_t();

    machines_t(scheduling_parameters_t param, weights_t weight);

    machines_t(machines_t &other);

    const std::vector<machine_t *> scheduledMachines();

    void addMachine(machine_t machine);

    void addPrescheduledJob(job_t *job);

    void addJobLocation(std::string lot_number,
                        std::vector<std::string> locations);
    void addJobProcessTimes(std::string lot_number,
                            std::map<std::string, double> uphs);

    void addGroupJobs(std::string recipe, std::vector<job_t *> jobs);

    void prescheduleJobs();

    std::string getModelByEntityName(std::string entity_name);

    void scheduleGroups();

    std::map<std::string, std::vector<std::string> > getModelLocations();

    ~machines_t();
};

inline std::map<std::string, std::vector<std::string> >
machines_t::getModelLocations()
{
    return _model_locations;
}

inline std::string machines_t::getModelByEntityName(std::string entity_name)
{
    return std::string(_machines.at(entity_name)->model_name.data.text);
}

inline const std::vector<machine_t *> machines_t::scheduledMachines()
{
    std::vector<machine_t *> machines;
    for (std::map<std::string, machine_t *>::iterator it = _machines.begin();
         it != _machines.end(); ++it) {
        if (it->second->base.size_of_jobs > 0) {
            machines.push_back(it->second);
        }
    }

    return machines;
}

inline void machines_t::addJobLocation(std::string lot_number,
                                       std::vector<std::string> locations)
{
    if (_job_can_run_locations.count(lot_number) != 0)
        std::cerr << "Warning : add job location twice, lot_number is ["
                  << lot_number << "]" << std::endl;

    _job_can_run_locations[lot_number] = locations;
}

inline void machines_t::addJobProcessTimes(
    std::string lot_number,
    std::map<std::string, double> process_times)
{
    if (_job_process_times.count(lot_number) != 0)
        std::cerr << "Warning : add job process_times twice, lot_number is ["
                  << lot_number << "]" << std::endl;

    _job_process_times[lot_number] = process_times;
}



#endif
