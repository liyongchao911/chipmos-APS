//
// Created by YuChunLin on 2021/11/20.
//
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#define private public
#define protected public

#include "include/machine_constraint.h"
#include "include/machine_constraint_r.h"

#undef private
#undef protected

using namespace std;

class test_machine_constraint_base_t : public testing::Test
{
protected:
    machine_constraint_t *mcs;

public:
    void SetUp() override;
};

void test_machine_constraint_base_t::SetUp()
{
    csv_t csv("ent_limit.csv", "r");
    csv.trim(" ");
    mcs = new machine_constraint_r_t(csv);
}