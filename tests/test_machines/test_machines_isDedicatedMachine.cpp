#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "include/infra.h"

#define private public
#define protected public

#include "include/job.h"
#include "include/machine.h"
#include "include/machines.h"

#undef private
#undef protected

using namespace std;

struct test_case_t {
    map<string, string> attr_input;
    map<string, bool> dedicated_machines_input;
    bool out;
};

struct test_case2_t {
    map<string, string> attr_input;
    map<string, bool> dedicated_machines_input;
    bool out;
};

class test_machines_is_dedicated_machine_t
    : public testing::TestWithParam<test_case2_t>
{
protected:
    machines_t *machines;
    // machine_t * machine;
    // job_t *job;

    void setupAttribute(machines_t *ms,
                        map<string, string> attr,
                        map<string, bool> d_m_input);

    void SetUp() override;
    void TearDown() override;
};

void test_machines_is_dedicated_machine_t::SetUp()
{
    machines = new machines_t();
}

void test_machines_is_dedicated_machine_t::TearDown()
{
    delete machines;
}

void test_machines_is_dedicated_machine_t::setupAttribute(
    machines_t *ms,
    map<string, string> attr,
    map<string, bool> d_m_input)
{
    if (attr["is_automotive"].compare("Y") == 0) {
        ms->_automotive_lot_numbers = set<string>{attr["lot_number"]};
    }

    for (auto it = d_m_input.begin(); it != d_m_input.end(); it++) {
        for (unsigned int i = 0; i < 100; ++i) {
            ms->_dedicate_machines[it->first]["BB"s + to_string(i)] =
                it->second;
            ms->_dedicate_machines["others"]["BB"s + to_string(i)] = it->second;
        }
    }
}


TEST_P(test_machines_is_dedicated_machine_t, test1)
{
    auto cs = GetParam();
    setupAttribute(machines, cs.attr_input, cs.dedicated_machines_input);
    EXPECT_EQ(machines->_isMachineDedicatedForJob(cs.attr_input["lot_number"],
                                                  cs.attr_input["cust"],
                                                  cs.attr_input["entity_name"]),
              cs.out);
}

INSTANTIATE_TEST_SUITE_P(
    test_is_dedicated_machine,
    test_machines_is_dedicated_machine_t,
    testing::Values(test_case2_t{{{"is_automotive"s, "Y"s},
                                  {"lot_number"s, "L1"s},
                                  {"cust"s, "C1"s},
                                  {"entity_name"s, "BB1"s}},
                                 {{"C1"s, true}},
                                 true},
                    test_case2_t{{{"is_automotive"s, "N"s},
                                  {"lot_number"s, "L1"s},
                                  {"cust"s, "C1"s},
                                  {"entity_name"s, "BB1"s}},
                                 {{"C1"s, true}},
                                 false},
                    test_case2_t{{{"is_automotive"s, "Y"s},
                                  {"lot_number"s, "L1"s},
                                  {"cust"s, "QQ"s},
                                  {"entity_name"s, "BB1"s}},
                                 {{"C1"s, true}},
                                 true}));
