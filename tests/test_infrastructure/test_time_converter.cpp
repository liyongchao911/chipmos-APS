#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "include/time_converter.h"

using namespace std;

struct string_to_time_test_case {
    string input;
    time_t ans;
};

ostream &operator<<(ostream &out, string_to_time_test_case cs)
{
    return out << "(" << cs.input << "," << cs.ans << ")";
}

class test_string_to_time_t
    : public testing::TestWithParam<string_to_time_test_case>
{
};

TEST_P(test_string_to_time_t, test_correctness)
{
    auto cs = GetParam();
    EXPECT_EQ(cs.ans, timeConverter()(cs.input))
        << "Failed with case : " << cs << endl;
}

INSTANTIATE_TEST_SUITE_P(
    string_to_time,
    test_string_to_time_t,
    testing::Values(

        string_to_time_test_case{"21-12-31 20:45", 1640954700},
        string_to_time_test_case{"22-03-03 03:16", 1646248560},
        string_to_time_test_case{"22-3-03 03:16", 1646248560},
        string_to_time_test_case{"22-03-3 03:16", 1646248560},
        string_to_time_test_case{"22-3-3 03:16", 1646248560},
        string_to_time_test_case{"20-02-29 03:16", 1582917360},
        string_to_time_test_case{"22-02-08 19:25", 1644319500},
        string_to_time_test_case{"22-02-08 19:25:01", 1644319501},
        string_to_time_test_case{"22-2-8 19:25:01", 1644319501},
        string_to_time_test_case{"22-02-8 19:25:01", 1644319501},
        string_to_time_test_case{"22-2-08 19:25:01", 1644319501},
        string_to_time_test_case{"2022/2/8 05:40:01", 1644270001},
        string_to_time_test_case{"2022/02/08 05:40:01", 1644270001},
        string_to_time_test_case{"2022/2/08 05:40:01", 1644270001},
        string_to_time_test_case{"2022/02/8 05:40:01", 1644270001},
        string_to_time_test_case{"2021/11/23 20:45", 1637671500},
        string_to_time_test_case{"2021/12/31 20:45", 1640954700},
        string_to_time_test_case{"2021/5/1 20:45", 1619873100},
        string_to_time_test_case{"2021/05/1 20:45", 1619873100},
        string_to_time_test_case{"2021/5/01 20:45", 1619873100},
        string_to_time_test_case{"2021/05/01 20:45", 1619873100},
        string_to_time_test_case{"", 0}));
