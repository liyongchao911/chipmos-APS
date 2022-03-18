#pragma once
#include <gtest/gtest.h>

#define private public
#define protected public
#include "include/route.h"
#undef private
#undef protected

using namespace std;

namespace test_route
{
class test_route_base_t : public testing::Test
{
protected:
    static csv_t *__routelist;

public:
    static route_t *route;

    static void SetUpTestSuite()
    {
        if (__routelist == nullptr) {
            test_route_base_t::__routelist =
                new csv_t("test_data/route_list.csv", "r", true, true);
            test_route_base_t::__routelist->setHeaders(
                {{"route", "wrto_route"},
                 {"oper", "wrto_oper"},
                 {"seq", "wrto_seq_num"},
                 {"desc", "wrto_opr_shrt_desc"}});
        }

        if (route == nullptr) {
            route = new route_t();
            route->setRoute(*__routelist);
        }
    }

    static void TearDownTestSuite()
    {
        if (__routelist != nullptr) {
            delete __routelist;
            __routelist = nullptr;
        }

        if (route != nullptr) {
            delete __routelist;
            __routelist = nullptr;
        }
    }
};



}  // namespace test_route
