#include "include/system_log.h"

void sys_log_t::output()
{
    FILE *file = fopen(_filename.c_str(), "w");
    fprintf(file, "Elapsed Time : %f sec\n", _elapsed_time);
    fprintf(file, "Wip information : \n");
    fprintf(file, "\tTotal wip : %d\n", _wip_total_number);
    fprintf(file, "\tnumber of scheduled lots : %d\n",
            _number_of_scheduled_jobs);
    fprintf(file, "\tnumber of unscheduled lots : %d\n",
            _number_of_unschedued_jobs);
    fprintf(file, "\tnumber of super hot run lots : %d\n",
            _number_of_super_hot_run_lots);
    fprintf(file, "Machine information : \n");
    fprintf(file, "\tNumber of available machines : %d\n",
            _number_of_available_machines);
    fprintf(file, "\tNumber of unavailable machines : %d\n",
            _number_of_unavailable_machines);
    fprintf(file, "Parent lots and their sublots : \n");
    for (auto it = _sublots.begin(); it != _sublots.end(); ++it) {
        fprintf(file, "\t%s : \n", it->first.c_str());
        for (unsigned int i = 0; i < it->second.size(); ++i) {
            fprintf(file, "\t\t%s\n", it->second[i].c_str());
        }
    }

    fprintf(file, "Cure time for a lot : \n");
    for (unsigned int i = 0; i < _cure_time_entries.size(); ++i) {
        if (_cure_time_entries[i].second - 0.0 > 0.00000001)
            fprintf(file, "\t%s : %f\n", _cure_time_entries[i].first.c_str(),
                    _cure_time_entries[i].second);
    }

    fprintf(file, "Prescheduled machines : \n");
    for (unsigned int i = 0; i < _prescheduled_machines.size(); ++i) {
        fprintf(file, "\t%s\n", _prescheduled_machines[i].c_str());
    }

    fprintf(file, "Number of tools : \n");
    for (auto it = _number_of_tools.begin(); it != _number_of_tools.end();
         ++it) {
        fprintf(file, "\t%s : %d\n", it->first.c_str(), it->second);
    }

    fclose(file);
}
