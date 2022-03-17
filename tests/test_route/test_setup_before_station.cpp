#include <gtest/gtest.h>

#include "include/csv.h"

#define private public
#define protected public
#include "include/route.h"
#undef private
#undef protected

#include <exception>
#include <map>
#include <string>
#include <vector>

using namespace std;

struct route_test_case_t {
    string test_route_name;
    vector<int> test_opers;
};



class test_route_t : public testing::TestWithParam<route_test_case_t>
{
public:
    route_t *route;
    csv_t __routelist;
    test_route_t() : __routelist("test_data/route_list.csv", "r", true, true)
    {
        __routelist.setHeaders({{"route", "wrto_route"},
                                {"oper", "wrto_oper"},
                                {"seq", "wrto_seq_num"},
                                {"desc", "wrto_opr_shrt_desc"}});
        route = new route_t();
        route->setRoute(__routelist);
    }

    ~test_route_t() { delete route; }
};


TEST_P(test_route_t, test_if_oper_in_WB_7)
{
    route_test_case_t cs = GetParam();
    set<int> route_list = route->_beforeWB[cs.test_route_name];
    vector<int> &test_oper = cs.test_opers;

    EXPECT_EQ(route_list.size(), test_oper.size());
    for (int i = 0; i < test_oper.size(); ++i) {
        EXPECT_EQ(route_list.count(test_oper[i]), 1);
    }
}

INSTANTIATE_TEST_SUITE_P(
    route_list,
    test_route_t,
    testing::Values(
        route_test_case_t{"BGA140",
                          {2050, 2060, 2061, 2070, 2080, 2150, 2090, 2200}},
        route_test_case_t{"BGA145",
                          {2040, 2050, 2060, 2070, 2080, 2150, 2090, 2200}},
        route_test_case_t{"BGA154",
                          {2050, 2060, 2061, 2070, 2080, 2150, 2090, 2200}},
        route_test_case_t{"BGA161",
                          {2040, 2050, 2060, 2070, 2080, 2150, 2090, 2200}},
        route_test_case_t{
            "MBGA138",
            {2070, 2080, 2150, 2090, 2200, 2205, 2209, 2130, 2140, 3150, 3200}},
        route_test_case_t{"MBGA176",
                          {2070, 2080, 2150, 2090, 2200, 2205, 2210, 2130, 3130,
                           3140, 3150, 3200}},
        route_test_case_t{"MBGA222",
                          {2070, 2130, 2080, 2150, 2090, 2200, 2205, 2210, 3130,
                           2140, 3330, 3340, 3150, 3200, 3205, 3210, 4130, 4330,
                           4340, 3350, 3400, 3205, 3210, 4130, 4330, 4340, 3350,
                           3400, 3405, 3410, 4530, 4730, 4740, 3550, 3600}}));
