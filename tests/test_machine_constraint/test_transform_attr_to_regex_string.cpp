//
// Created by YuChunLin on 2021/11/20.
//

#include <gtest/gtest.h>
#include <regex>

#define private public
#define protected public

#include "include/machine_constraint.h"

#undef private
#undef protected

using namespace std;

struct to_regex_test_case_t {
    string input;
    string out;
};

ostream &operator<<(ostream &out, struct to_regex_test_case_t cs)
{
    return out << "out : " << cs.out;
}

class test_transformPkgIdToRegex_t
    : public testing::TestWithParam<to_regex_test_case_t>
{
};

TEST_P(test_transformPkgIdToRegex_t, test_pkg_regex)
{
    auto cs = GetParam();
    string re = machine_constraint_t::transformPkgIdToRegex(cs.input);
    EXPECT_TRUE(re == cs.out) << "re = " << re;
}

INSTANTIATE_TEST_SUITE_P(
    test_pkg_regex,
    test_transformPkgIdToRegex_t,
    testing::Values(to_regex_test_case_t{"*56*"s, R"(\w*56\w*)"s},
                    to_regex_test_case_t{"**5*6**s"s, R"(\w*\w*5\w*6\w*\w*s)"s},
                    to_regex_test_case_t{"12{FS}34"s, R"(12\{FS\}34)"s},
                    to_regex_test_case_t{"12*{FS}*34"s,
                                         R"(12\w*\{FS\}\w*34)"s}));


class test_transformStarToRegexString
    : public testing::TestWithParam<to_regex_test_case_t>
{
};

TEST_P(test_transformStarToRegexString, test_star_regex)
{
    auto cs = GetParam();
    string re = machine_constraint_t::transformStarToRegexString(cs.input);
    EXPECT_TRUE(re == cs.out);
}

INSTANTIATE_TEST_SUITE_P(
    test_star_regex,
    test_transformStarToRegexString,
    testing::Values(to_regex_test_case_t{"*1*"s, R"(\w*1\w*)"s},
                    to_regex_test_case_t{"*1D**F*"s, R"(\w*1D\w*\w*F\w*)"s}));