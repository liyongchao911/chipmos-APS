#include <gtest/gtest.h>

#include "include/csv.h"
#include "tests/test_route/test_route.h"

#define private public
#define protected public
#include "include/lot.h"
#include "include/lot_base.h"
#include "include/route.h"
#undef private
#undef protected

#include <iostream>
#include <string>

using namespace std;


namespace route_index
{


struct route_index_test_case_t {
    string test_route_name;
    int test_oper;
    int rs_idx;

    friend ostream &operator<<(ostream &out,
                               const struct route_index_test_case_t &cs)
    {
        return out << "(" << cs.test_route_name << ", " << cs.test_oper << ", "
                   << cs.rs_idx << ")";
    }
};

class test_route_idx_t
    : public testing::WithParamInterface<struct route_index_test_case_t>,
      public ::test_route::test_route_base_t
{
};


TEST_P(test_route_idx_t, test_index)
{
    auto cs = GetParam();
    EXPECT_EQ(route->findStationIdx(cs.test_route_name, cs.test_oper),
              cs.rs_idx);
}

INSTANTIATE_TEST_SUITE_P(
    oper_index,
    test_route_idx_t,
    testing::Values(route_index_test_case_t{"BGA140", 2050, 0},
                    route_index_test_case_t{"BGA140", 2000, -1},
                    route_index_test_case_t{"MBGA222", 2070, 0},
                    route_index_test_case_t{"MBGA222", 2130, 1},
                    route_index_test_case_t{"MBGA222", 2080, 2},
                    route_index_test_case_t{"MBGA222", 3130, 8},
                    route_index_test_case_t{"MBGA222", 4330, 17},
                    route_index_test_case_t{"MBGA222", 3350, 19},
                    route_index_test_case_t{"MBGA222", 3600, 27}

                    ));

}  // namespace route_index
