#include <gtest/gtest.h>

#include <string>

#include "tests/test_route/test_route.h"

using namespace std;

namespace cure_time
{
struct cure_time_test_case_t {
    string process_id;
    int oper;
    int expected_cure_time;

    friend ostream &operator<<(ostream &os,
                               const struct cure_time_test_case_t &cs)
    {
        return os << "(" << cs.process_id << "," << cs.oper << ")->"
                  << cs.expected_cure_time << "";
    }
};

class test_get_cure_time_t
    : public ::testing::WithParamInterface<struct cure_time_test_case_t>,
      public ::test_route::test_route_base_t
{
};

TEST_P(test_get_cure_time_t, get_cure_time)
{
    auto cs = GetParam();
    EXPECT_EQ(route->getCureTime(cs.process_id, cs.oper),
              cs.expected_cure_time);
}

INSTANTIATE_TEST_SUITE_P(
    cure_time,
    test_get_cure_time_t,
    testing::Values(cure_time_test_case_t{"0008W999", 2080, 90},
                    cure_time_test_case_t{"0008W999", 2150, 120},
                    cure_time_test_case_t{"0008W999", 2050, 0}));
}  // namespace cure_time
