#include <gtest/gtest.h>

#define protected public
#define private public

#include "include/machines.h"

#undef proctected
#undef private

#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

class test_machines_groupBy : public testing::Test
{
private:
    vector<string> part_ids1;
    vector<string> part_ids2;

    vector<string> part_nos1;
    vector<string> part_nos2;


    vector<job_t *> all_jobs;

    map<string, struct __machine_group_t *> _groups;

public:
    machines_t *machines;
    void SetUp() override;
    void TearDown() override;
    vector<job_t *> createJobs(vector<string> part_id, vector<string> part_no);
};

vector<job_t *> test_machines_groupBy::createJobs(vector<string> part_id,
                                                  vector<string> part_no)
{
    vector<job_t *> jobs;
    for (unsigned int i = 0; i < part_id.size(); ++i) {
        for (unsigned int j = 0; j < part_no.size(); ++j) {
            job_t *job = new job_t();
            job->part_no = stringToInfo(part_no[j]);
            job->part_id = stringToInfo(part_id[i]);
            jobs.push_back(job);
            all_jobs.push_back(job);
        }
    }
    return jobs;
}

void test_machines_groupBy::SetUp()
{
    part_ids1 = vector<string>({"PART_ID1_1", "PART_ID1_2"});
    part_nos1 = vector<string>({"PART_NO1_1", "PART_NO1_2"});

    part_ids2 = vector<string>({"PART_ID2_1", "PART_ID2_2"});
    part_nos2 = vector<string>({"PART_NO2_1", "PART_NO2_2"});

    _groups["1_1"] = new (struct __machine_group_t);
    _groups["1_2"] = new (struct __machine_group_t);
    _groups["2_1"] = new (struct __machine_group_t);
    _groups["1_2"] = new (struct __machine_group_t);

    *_groups["1_1"] = __machine_group_t{
        .unscheduled_jobs = createJobs(part_ids1, part_nos1),
    };

    *_groups["1_2"] = __machine_group_t{
        .unscheduled_jobs = createJobs(part_ids1, part_nos2),
    };

    *_groups["2_1"] = __machine_group_t{
        .unscheduled_jobs = createJobs(part_ids2, part_nos1),
    };

    *_groups["2_2"] = __machine_group_t{
        .unscheduled_jobs = createJobs(part_ids2, part_nos2),
    };

    machines = new machines_t();
    machines->_dispatch_groups = _groups;
}

void test_machines_groupBy::TearDown()
{
    delete machines;
    for (unsigned int i = 0; i < all_jobs.size(); ++i)
        delete all_jobs[i];
}

// TEST_F(test_machines_groupBy, test_groupBy)
// {
//     machines->groupJobsByToolAndWire();
//     for (unsigned int bd = 1; bd <= 2; ++bd) {
//         for (unsigned int i = 1; i <= 2; ++i) {  // part_id
//             string part_id = "PART_ID" + to_string(bd) + "_" + to_string(i);
//             for (unsigned int j = 1; j <= 2; ++j) {  // part_no
//                 string part_no = "PART_NO" + to_string(bd) + "_" +
//                 to_string(j); string key = part_no + "_" + part_id;
//
//                 EXPECT_EQ(machines->_tool_wire_jobs_groups.count(key), 1);
//                 EXPECT_EQ(
//                     machines->_tool_wire_jobs_groups[key]->orphan_jobs.size(),
//                     1);
//                 EXPECT_EQ(machines->_tool_jobs_groups.count(part_no), 1);
//                 EXPECT_EQ(machines->_tool_jobs_groups[part_no].size(), 4);
//             }
//             EXPECT_EQ(machines->_wire_jobs_groups.count(part_id), 1);
//             EXPECT_EQ(machines->_wire_jobs_groups[part_id].size(), 4);
//         }
//     }
// }
