#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#define private public
#define protected public
#include "include/lot_base.h"
#undef private
#undef protected

using namespace std;

struct setupDefaultValueRow_test_case {
    map<string, string> input;
};

class test_lot_base_setupDefaultValueOfRow_t
    : public testing::TestWithParam<setupDefaultValueRow_test_case>
{
public:
    lot_base_t *lot_base;
    void SetUp() override;
    void TearDown() override;
};

void test_lot_base_setupDefaultValueOfRow_t::SetUp()
{
    lot_base = new lot_base_t();
}

void test_lot_base_setupDefaultValueOfRow_t::TearDown()
{
    delete lot_base;
}
/**
 * In this test case, test if all default keys are set
 */
TEST_P(test_lot_base_setupDefaultValueOfRow_t,
       test_setupDefaultValueOfLotBase_default_keys)
{
    auto cs = GetParam();
    map<string, string> dup_input = cs.input;
    lot_base->_setupDefaultValueOfRow(dup_input);
    // Test all key is in dup_input
    map<string, vector<string> > default_values =
        lot_base->getRowDefaultValues();
    for (auto it = default_values.begin(); it != default_values.end(); it++) {
        for (unsigned int i = 0; i < it->second.size(); ++i) {
            EXPECT_EQ(dup_input.count(it->second[i]), 1);
        }
    }
}

/**
 * In this test case, test if all default values are set
 */
TEST_P(test_lot_base_setupDefaultValueOfRow_t,
       test_setupDefaultValueOfLotBase_default_values)
{
    auto cs = GetParam();
    map<string, string> dup_input = cs.input;
    lot_base->_setupDefaultValueOfRow(dup_input);
    // Test all key is in dup_input
    map<string, vector<string> > default_values =
        lot_base->getRowDefaultValues();
    for (auto it = default_values.begin(); it != default_values.end(); it++) {
        for (unsigned int i = 0; i < it->second.size(); ++i) {
            EXPECT_NO_THROW(dup_input.at(it->second[i]));
            if (cs.input.count(it->second[i]) == 0)
                EXPECT_EQ(dup_input.at(it->second[i]), it->first);
        }
    }
}

/**
 * Test if the function doesn't remove unrelated keys
 */
TEST_P(test_lot_base_setupDefaultValueOfRow_t,
       test_setupDefaultValueOfLotBase_unused_keys)
{
    auto cs = GetParam();
    map<string, string> dup_input = cs.input;
    lot_base->_setupDefaultValueOfRow(dup_input);
    // Test all key is in dup_input
    map<string, vector<string> > default_values =
        lot_base->getRowDefaultValues();
    for (auto it = cs.input.begin(); it != cs.input.end(); it++) {
        EXPECT_EQ(dup_input.count(it->first), 1);
    }
}

INSTANTIATE_TEST_SUITE_P(setupDefaultValueRow,
                         test_lot_base_setupDefaultValueOfRow_t,
                         testing::Values(setupDefaultValueRow_test_case{{}},
                                         setupDefaultValueRow_test_case{
                                             {{"lot_number", "RXXLLL"},
                                              {"pin_package", "PINPKG"},
                                              {"Hello", "hello"}}}));
