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
                    cure_time_test_case_t{"000IA119", 2022, 210},
                    cure_time_test_case_t{"0086A095", 2140, 40},
                    cure_time_test_case_t{"000IAE07ENG", 2022, 210},
                    cure_time_test_case_t{"0086A124", 2140, 60},
                    cure_time_test_case_t{"008OK001", 2250, 90},
                    cure_time_test_case_t{"008YM141", 2150, 90},
                    cure_time_test_case_t{"008YQ170", 2250, 30},
                    cure_time_test_case_t{"048XZ001", 3140, 75},
                    cure_time_test_case_t{"008YD009ENG", 2250, 30},
                    cure_time_test_case_t{"060CD121", 2080, 90},
                    cure_time_test_case_t{"0345V009", 2080, 100},
                    cure_time_test_case_t{"060U9005", 2080, 70},
                    cure_time_test_case_t{"096EL063", 2080, 80},
                    cure_time_test_case_t{"096EL095", 2080, 0},
                    cure_time_test_case_t{"100X6006", 2051, 90},
                    cure_time_test_case_t{"153XT081", 2250, 120},
                    cure_time_test_case_t{"008YD009ENG", 2225, 90},
                    cure_time_test_case_t{"008DR010", 2330, 30},
                    cure_time_test_case_t{"008YQ085", 2405, 90},
                    cure_time_test_case_t{"008FG005", 2425, 90},
                    cure_time_test_case_t{"008YD088", 2428, 90},
                    cure_time_test_case_t{"345XT066", 5340, 90},
                    cure_time_test_case_t{"156KP004ENG", 4140, 40},

                    cure_time_test_case_t{"0008W999", 2050, 0},
                    cure_time_test_case_t{"000IA066", 2425, 0},
                    cure_time_test_case_t{"0086A123", 2070, 0},
                    cure_time_test_case_t{"0225S001", 2405, 0},
                    cure_time_test_case_t{"0442J030", 5340, 0},
                    cure_time_test_case_t{"048TG295", 2050, 0},
                    cure_time_test_case_t{"048TV269", 2472, 0},
                    cure_time_test_case_t{"054TS031", 3140, 0},
                    cure_time_test_case_t{"096EC002", 2428, 0},
                    cure_time_test_case_t{"100X6E001ENG", 2051, 0}));
}  // namespace cure_time
