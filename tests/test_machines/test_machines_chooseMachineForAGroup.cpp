#include <gtest/gtest.h>

#define private public
#define protected public

#include "include/machines.h"

#undef private
#undef protected

#include <map>
#include <string>

using namespace std;

class test_machines_t_chooseMachinesForAGroup : public testing::Test
{
private:
public:
    map<string, string> machine_case1;
    map<string, string> machine_case2;
    map<string, string> machine_case3;

    map<string, string> job_case1;
    map<string, string> job_case2;

    map<string, string> group_config1;
    map<string, string> group_config2;

    machines_t *machines;

    struct __job_group_t *g1;
    struct __job_group_t *g2;

    vector<job_t *> createJobs(map<string, string> cfg, int number, int idx);
    vector<machine_t *> createMachines(map<string, string> cfg, int number);

    void createJobGroup(struct __job_group_t **g,
                        map<int, map<string, string> > job_config,
                        map<int, map<string, string> > machine_config,
                        map<string, string> group_config);

    void SetUp() override;
    void TearDown() override;
};

vector<job_t *> test_machines_t_chooseMachinesForAGroup::createJobs(
    map<string, string> cfg,
    int number,
    int idx)
{
    vector<job_t *> jobs;
    job_t *job;
    for (int i = 0; i < number; ++i) {
        string lot_number(cfg["lot_number"] + "_" + to_string(idx + i));
        job = new job_t();
        *job = job_t{.base.job_info = stringToInfo(cfg[lot_number]),
                     .part_no = stringToInfo(cfg["part_no"]),
                     .part_id = stringToInfo(cfg["part_id"])};
    }
}

void test_machines_t_chooseMachinesForAGroup::SetUp()
{
    machine_case1 =
        map<string, string>({{"model", "UTC1000"}, {"location", "TA"}});

    machine_case2 =
        map<string, string>({{"model", "UTC1000"}, {"location", "TB"}});

    machine_case3 =
        map<string, string>({{"model", "UTC3000"}, {"location", "TB"}});

    job_case1 = map<string, string>({{"lot_number", "J1"},
                                     {"location", "TA,TB"},
                                     {"model", "UTC1000,UTC3000"},
                                     {"part_id", "PART_ID1"},
                                     {"part_no", "PART_NO1"}});

    job_case2 = map<string, string>({{"lot_number", "J2"} {"location", "TB"},
                                     {"model", "UTC3000"},
                                     {"part_id", "PART_ID2"},
                                     {"part_no", "PART_NO2"}});
}