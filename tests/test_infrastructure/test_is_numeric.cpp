#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "include/infra.h"

using namespace std;

namespace is_numeric
{
struct is_numeric_test_data_t {
    string text;
    bool res;
    friend ostream &operator<<(ostream &out,
                               const struct is_numeric_test_data_t &data)
    {
        return out << "(" << data.text << ", " << (data.res ? "true" : "false")
                   << ")";
    }
};

class test_is_numeric_t
    : public testing::TestWithParam<struct is_numeric_test_data_t>
{
};

// TEST_P(test_is_numeric_t, is_numeric_func)
// {
//     auto cs = GetParam();
//     EXPECT_EQ(isNumeric(cs.text), cs.res);
// }

TEST_P(test_is_numeric_t, is_numeric_obj)
{
    auto cs = GetParam();
    EXPECT_EQ(isNumeric()(cs.text), cs.res);
}

INSTANTIATE_TEST_SUITE_P(
    isNumeric,
    test_is_numeric_t,
    testing::Values(is_numeric_test_data_t{"123", true},
                    is_numeric_test_data_t{"123.0", true},
                    is_numeric_test_data_t{"12.3", true},
                    is_numeric_test_data_t{".123", true},
                    is_numeric_test_data_t{"12..3", false},
                    is_numeric_test_data_t{"1.2.3", false},
                    is_numeric_test_data_t{"-123", true},
                    is_numeric_test_data_t{"-123.0", true},
                    is_numeric_test_data_t{"-12.3", true},
                    is_numeric_test_data_t{"-.123", true},
                    is_numeric_test_data_t{"-12..3", false},
                    is_numeric_test_data_t{"-1.2.3", false},
                    is_numeric_test_data_t{".2", true},
                    is_numeric_test_data_t{"-2.", true},
                    is_numeric_test_data_t{"-.", false},
                    is_numeric_test_data_t{"-.2", true}));

}  // namespace is_numeric
