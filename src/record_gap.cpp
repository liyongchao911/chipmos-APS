#include <algorithm>

#include "include/record_gap.h"

Record_gap::Record_gap(machine_base_operations_t *op,
                       std::string directory,
                       double th)
    : threshold(th),
      nfp{
          __SETUP(CWN), __SETUP(CK),  __SETUP(EU),  __SETUP(MC),
          __SETUP(SC),  __SETUP(CSC), __SETUP(CSC),
      },
      ops(op),
      csv_file(directory + "/record_gap.csv", "w")
{
}

double Record_gap::calculateSetupTime(job_t *prev,
                                      job_t *next,
                                      machine_base_operations_t *ops)
{
    str.clear();
    bool first = true;

    double time = 0;
    if (isSameInfo(prev->bdid, next->bdid)) {
        return 0.0;
    }
    for (unsigned int i = 0; i < ops->sizeof_setup_time_function_array; ++i) {
        double prev_time = time;
        time += ops->setup_time_functions[i].function(
            &prev->base, &next->base, ops->setup_time_functions[i].minute);
        if (time - prev_time >= threshold) {
            if (first) {
                str += nfp[i].name;
                first = false;
            } else {
                str += ",";
                str += nfp[i].name;
            }
        }
    }

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
        double setup_time = calculateSetupTime(*it, *std::prev(it), ops);
        double arrival_time_gap = (*it)->base.arriv_t - (*it)->base.start_time;

        double Ts = (*std::prev(it))->base.end_time;
        double Tw = Ts + setup_time;
        double Ta = (*it)->base.arriv_t;
        double Tr = (*it)->base.start_time;

        if (setup_time >= threshold)
            csv_file.addData(std::map<std::string, std::string>({
                {"0number", std::string((*it)->base.machine_no.data.text) +
                                "_" + std::to_string(count)},
                {"1entity", std::string((*it)->base.machine_no.data.text)},
                {"2jobcode", str},
                {"3lot_num_1",
                 std::string((*std::prev(it))->base.job_info.data.text)},
                {"4lot_num_2", std::string((*(it))->base.job_info.data.text)},
                {"5bd_id_1", std::string((*std::prev(it))->bdid.data.text)},
                {"6bd_id_2", std::string((*(it))->bdid.data.text)},
                {"7start_time", std::to_string(Tw)},
                {"8end_time", std::to_string(Tr)},
            }));
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
    for (const auto &it : _jobs)
        record_gap_single_machine(it.second);
    csv_file.write();
}
