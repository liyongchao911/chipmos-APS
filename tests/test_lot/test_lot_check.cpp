#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

#include <gtest/gtest.h>

#define private public
#define protected public

#include "include/lot.h"

#undef private
#undef protected

using namespace std;

struct check_test_case_t {
    map<string, string> elements;
    bool out;
};

ostream &operator<<(ostream &out, const check_test_case_t &cs)
{
    // out << cs.elements;
    for (auto it = cs.elements.begin(); it != cs.elements.end(); it++) {
        out << "{" << it->first << " : " << it->second << "}" << endl;
    }
    return out;
}

class test_lot_check_data_format_t
    : public testing::TestWithParam<check_test_case_t>
{
protected:
    lot_t *lot;
    check_test_case_t cs;
    test_lot_check_data_format_t() { lot = new lot_t(); }
    ~test_lot_check_data_format_t() { delete lot; }
};


TEST_P(test_lot_check_data_format_t, check_data_format)
{
    check_test_case_t cs = GetParam();
    string log;
    EXPECT_EQ(lot->checkDataFormat(cs.elements, log), cs.out);
}

INSTANTIATE_TEST_SUITE_P(
    check_data_format,
    test_lot_check_data_format_t,
    testing::Values(
        check_test_case_t{{{{"queue_time"s, "0"s}}}, true},
        check_test_case_t{{{{"fcst_time"s, "0"s}}}, true},
        check_test_case_t{{{{"dest_oper"s, "0"s}}}, true},
        check_test_case_t{{{{"amount_of_tools"s, "0"s}}}, true},
        check_test_case_t{{{{"amount_of_wires"s, "0"s}}}, true},
        check_test_case_t{{{{"sub_lot"s, "0"s}}}, true},
        check_test_case_t{{{{"qty"s, "0"s}}}, true},

        check_test_case_t{{{{"queue_time"s, "0"s}, {"fcst_time"s, ""s}}},
                          false},
        check_test_case_t{{{{"fcst_time"s, "0"s}, {"queue_time"s, "0"s}}},
                          true},
        check_test_case_t{{{{"queue_time"s, "0"s},
                            {"amount_of_wires"s, "0"s},
                            {"fcst_time"s, ""s}}},
                          false},

        check_test_case_t{{{{"queue_time"s, ""s}}}, false},
        check_test_case_t{{{{"fcst_time"s, ""s}}}, false},
        check_test_case_t{{{{"dest_oper"s, ""s}}}, false},
        check_test_case_t{{{{"amount_of_tools"s, ""s}}}, false},
        check_test_case_t{{{{"amount_of_wires"s, "78.adf"s}}}, false},
        check_test_case_t{{{{"sub_lot"s, "gsl.123"s}}}, false},
        check_test_case_t{{{{"qty"s, ""s}}}, false}

        ));
