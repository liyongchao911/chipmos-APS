#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#define private public
#define protected public

#include "include/lot.h"

#undef private
#undef protected

using namespace std;

struct addLog_test_case_t {
    vector<ERROR_T> log_err;
    vector<ERROR_T> out;
};

class test_lot_add_log_t : public testing::TestWithParam<addLog_test_case_t>
{
protected:
    lot_t *lot;

    test_lot_add_log_t() { lot = new lot_t(); }
    ~test_lot_add_log_t() { delete lot; }
};

TEST_P(test_lot_add_log_t, test_add_log)
{
    addLog_test_case_t cs = GetParam();
    for (auto err : cs.log_err) {
        lot->addLog("", err);
    }

    EXPECT_EQ(lot->_statuses.size(), cs.out.size());
    for (int i = 0, size = lot->_statuses.size(); i < size; ++i) {
        EXPECT_EQ(lot->_statuses[i], cs.out[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    test_add_log,
    test_lot_add_log_t,
    testing::Values(addLog_test_case_t{{ERROR_BAD_DATA_FORMAT, ERROR_BOM_ID},
                                       {ERROR_BAD_DATA_FORMAT, ERROR_BOM_ID}},
                    addLog_test_case_t{
                        {ERROR_BAD_DATA_FORMAT, SUCCESS, ERROR_BOM_ID},
                        {ERROR_BAD_DATA_FORMAT, ERROR_BOM_ID}},  // skip SUCCESS
                    addLog_test_case_t{
                        {ERROR_BAD_DATA_FORMAT, SUCCESS, SUCCESS, ERROR_BOM_ID},
                        {ERROR_BAD_DATA_FORMAT, ERROR_BOM_ID}},  // skip SUCCESS
                    addLog_test_case_t{{SUCCESS, SUCCESS, SUCCESS, SUCCESS}, {}}
                    // skip SUCCESS
                    ));
