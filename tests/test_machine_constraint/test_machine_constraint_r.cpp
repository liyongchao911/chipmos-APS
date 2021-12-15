//
// Created by YuChunLin on 2021/11/22.
//

#include <gtest/gtest.h>
#include <string>

#define private public
#define protected public

#include "include/machine_constraint.h"
#include "include/machine_constraint_r.h"

#undef private
#undef protected

using namespace std;

struct testing_machine_constraint_r_case {
    string entity_regex_str;
    string restrained_model;
    string entity_name;
    string model_name;
    bool out;
};

class test_machine_constraint_r_t
    : public testing::TestWithParam<testing_machine_constraint_r_case>
{
protected:
    machine_constraint_r_t *mcs_r;

    void SetUp() override;
    void TearDown() override;
};

void test_machine_constraint_r_t::SetUp()
{
    mcs_r = new machine_constraint_r_t();
}

void test_machine_constraint_r_t::TearDown()
{
    delete mcs_r;
}

TEST_P(test_machine_constraint_r_t, test_is_machine_restrained)
{
    auto cs = GetParam();
    regex re(cs.entity_regex_str);
    bool ret_val = mcs_r->_isMachineRestrained(re, cs.restrained_model,
                                               cs.entity_name, cs.model_name);
    EXPECT_EQ(cs.out, ret_val);
}

INSTANTIATE_TEST_SUITE_P(
    machine_constraint_r,
    test_machine_constraint_r_t,
    testing::Values(testing_machine_constraint_r_case{R"(BB\w*)"s, "UTC3000"s,
                                                      "BB556"s, "UTC3000"s,
                                                      false},
                    testing_machine_constraint_r_case{
                        R"(BB5\w*)"s, "UTC3000"s, "BB656"s, "UTC3000"s, true},
                    testing_machine_constraint_r_case{
                        R"(BB5\w*)"s, "UTC2000"s, "BB556"s, "UTC3000"s, true}));