#ifndef __SYSTEM_LOG_H__
#define __SYSTEM_LOG_H__

#include <map>
#include <string>
#include <vector>

#include "include/lots.h"
#include "include/route.h"

class sys_log_t
{
protected:
    double _elapsed_time;
    int _wip_total_number, _number_of_scheduled_jobs,
        _number_of_unschedued_jobs, _number_of_total_machines,
        _number_of_available_machines, _number_of_unavailable_machines,
        _number_of_super_hot_run_lots;
    // _number_of_prescheudled_machines,
    // _number_of_nonprescheduled_machines;

    std::string _filename;
    std::map<std::string, std::vector<std::string> > _sublots;
    std::map<std::string, int> _number_of_tools;
    std::vector<std::pair<std::string, double> > _cure_time_entries;
    std::vector<std::string> _prescheduled_machines;

public:
    sys_log_t(std::string filename) : _filename(filename)
    {
        _elapsed_time = _wip_total_number = _number_of_scheduled_jobs =
            _number_of_unschedued_jobs = _number_of_total_machines =
                _number_of_available_machines =
                    _number_of_unavailable_machines =
                        _number_of_super_hot_run_lots
            // = _number_of_prescheudled_machines
            // = _number_of_nonprescheduled_machines
            = 0;
    }

    inline void setSysTimeElapse(double t) { _elapsed_time = t; }
    inline void setWipTotalNumber(int number) { _wip_total_number = number; }
    inline void setNumberOfScheduledJobs(int number)
    {
        _number_of_scheduled_jobs = number;
    }
    inline void setNumberOfUnscheduledJobs(int number)
    {
        _number_of_unschedued_jobs = number;
    }
    inline void setNumberOfTotalMachine(int number)
    {
        _number_of_total_machines = number;
    }
    inline void setNumberOfAvailableMachines(int number)
    {
        _number_of_available_machines = number;
    }
    inline void setNumberOfUnavailableMachines(int number)
    {
        _number_of_unavailable_machines = number;
    }
    inline void setSublot(
        std::map<std::string, std::vector<std::string> > sublots)
    {
        _sublots = sublots;
    }

    inline void setNumberOfSuperHotRunLots(int number)
    {
        _number_of_super_hot_run_lots = number;
    }
    inline void setNumberOfTools(std::map<std::string, int> number_of_tools)
    {
        _number_of_tools = number_of_tools;
    }
    inline void setPrescheduledMachines(std::vector<std::string> entities)
    {
        _prescheduled_machines = entities;
    }
    // inline void setNumberOfPrescheduledMachines(int number) {
    // _number_of_prescheudled_machines = number; } inline void
    // setNumberOfNonprescheduledMachines(int number) {
    // _number_of_nonprescheduled_machines = number; }
    inline void setOutputFileName(std::string filename)
    {
        _filename = filename;
    }
    inline void setCureTimeForSingleLot(
        std::pair<std::string, double> cure_time)
    {
        _cure_time_entries.push_back(cure_time);
    }
    virtual void output();
    virtual ~sys_log_t() {}
};

#endif
