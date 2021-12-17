#include <algorithm>

#include "include/record_gap.h"

Record_gap::Record_gap(machine_base_operations_t *op, double th)
    : threshold(th),
      nfp{
          __SETUP(CWN), __SETUP(CK),  __SETUP(EU),  __SETUP(MC),
          __SETUP(SC),  __SETUP(CSC), __SETUP(CSC),
      },
      ops(op)
{
}

double Record_gap::calculateSetupTime(job_t *prev,
                                      job_t *next,
                                      machine_base_operations_t *ops)
{
    str.clear();
    str += "\"";

    double time = 0;
    if (isSameInfo(prev->bdid, next->bdid)) {
        str += "\"";
        return 0.0;
    }
    for (unsigned int i = 0; i < ops->sizeof_setup_time_function_array; ++i) {
        double prev_time = time;
        time += ops->setup_time_functions[i].function(
            &prev->base, &next->base, ops->setup_time_functions[i].minute);
        if (time - prev_time >= threshold) {
            if (str.back() == '"')
                str += nfp[i].name;
            else {
                str += ",";
                str += nfp[i].name;
            }
        }
    }

    str += "\"";
    return time;
}

void Record_gap::record_gap_single_machine(std::vector<job_t *> jobs)
{
    std::sort(jobs.begin(), jobs.end(), [](job_t *j1, job_t *j2) {
        return j1->base.start_time < j2->base.start_time;
    });

    int count = 1;
    for (std::vector<job_t *>::iterator it = std::next(jobs.begin());
         it != jobs.end(); ++it, ++count) {
        double setup_time = calculateSetupTime(*it, *(it - 1), ops);
        double arrival_time_gap = (*it)->base.arriv_t - (*it)->base.start_time;

        double Ts = (*(it - 1))->base.end_time;
        double Tw = Ts + setup_time;
        double Ta = (*it)->base.arriv_t;
        double Tr = (*it)->base.start_time;

        outputFile << (*it)->base.machine_no.data.text << '_' << count << ','
                   << (*it)->base.machine_no.data.text << ',' << str
                   << (*(it - 1))->base.job_info.data.text << ','
                   << (*(it))->base.job_info.data.text << ','
                   << (*(it - 1))->bdid.data.text << ','
                   << (*(it))->bdid.data.text << ',' << Tw << ',' << Tr
                   << std::endl;
    }
}

void Record_gap::addJob(job_t *job)
{
    std::map<std::string, std::vector<job_t *>>::iterator it;
    std::string machine_name((job->base).machine_no.data.text);
    _jobs[machine_name].push_back(job);
}

void Record_gap::record_gap_all_machines()
{
    outputFile.open("out.csv");

    outputFile << "number" << ',' << "entity" << ',' << "jobcode" << ','
               << "lot_num_1" << ',' << "lot_num_2" << ',' << "bd_id_1" << ','
               << "bd_id_2" << ',' << "start_time" << ',' << "end_time" << ','
               << std::endl;

    for (const auto &it : _jobs)
        record_gap_single_machine(it.second);
    outputFile.close();
}
