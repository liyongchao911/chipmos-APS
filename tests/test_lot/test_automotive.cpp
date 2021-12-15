#include <gtest/gtest.h>
#include <iostream>
#include <string>

using namespace std;

#define private public
#define protected public

#include "include/lot.h"

#undef private
#undef protected

using namespace std;

struct test_case_t {
    string automotive_val;
    bool out;
};

class test_lot_automotive_t : public testing::TestWithParam<test_case_t>
{
protected:
    lot_t *lot;
    test_lot_automotive_t() { lot = new lot_t(); }
    ~test_lot_automotive_t() { delete lot; }
};

TEST_P(test_lot_automotive_t, test_lot_automotive)
{
    auto cs = GetParam();
    lot->setAutomotive(cs.automotive_val);
    EXPECT_EQ(lot->isAutomotive(), cs.out);
}

INSTANTIATE_TEST_SUITE_P(test_lot_automotive,
                         test_lot_automotive_t,
                         testing::Values(test_case_t{"Y"s, true},
                                         test_case_t{"N"s, false},
                                         test_case_t{""s, false},
                                         test_case_t{" "s, false}));
