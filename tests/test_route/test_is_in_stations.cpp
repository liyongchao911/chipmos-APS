#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <iostream>

#define private public
#define protected public
#include "include/lot.h"
#undef private
#undef protected

#include "tests/test_route/test_route.h"

namespace test_route_is_in_stations
{

struct lot_status_test_case_t {
    string test_route_name;
    int test_oper;
    bool is_mvin;
    bool rs;

    friend ostream &operator<<(ostream &out,
                               const struct lot_status_test_case_t &status)
    {
        return out << "(" << status.test_oper
                   << ", mvin :" << (status.is_mvin ? "true" : "false")
                   << ", ans :" << status.rs << ")";
    }
};

class test_route_is_in_stations_t
    : public testing::WithParamInterface<lot_status_test_case_t>,
      public test_route::test_route_base_t
{
};

TEST_P(test_route_is_in_stations_t, test_is_in_stations)
{
    auto cs = GetParam();
    lot_t *lot = new lot_t();
    lot->_oper = cs.test_oper;
    lot->_route = cs.test_route_name;
    lot->_mvin = cs.is_mvin;

    EXPECT_EQ(route->isLotInStations(*lot), cs.rs);
}

INSTANTIATE_TEST_SUITE_P(
    is_in_stations,
    test_route_is_in_stations_t,
    testing::Values(lot_status_test_case_t{"BGA140", 2050, true, true},
                    lot_status_test_case_t{"BGA140", 2200, true, false},
                    lot_status_test_case_t{"BGA140", 2200, false, true},
                    lot_status_test_case_t{"MBGA222", 3200, true, true},
                    lot_status_test_case_t{"MBGA222", 3600, true, true},
                    lot_status_test_case_t{"MBGA222", 3600, false, true},
                    lot_status_test_case_t{"MBGA222", 4200, false, true},
                    lot_status_test_case_t{"MBGA222", 4200, true, false}));

}  // namespace test_route_is_in_stations
