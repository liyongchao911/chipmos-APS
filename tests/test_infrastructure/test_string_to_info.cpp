//
// Created by YuChunLin on 2021/11/22.
//

#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "include/infra.h"

using namespace std;

struct string_to_info_test_case {
    string input;
    unsigned int text_length;
    unsigned int number_size;
};

ostream &operator<<(ostream &out, string_to_info_test_case cs)
{
    return out << "(" << cs.input << ", " << cs.text_length << ", "
               << cs.number_size << ")";
}

ostream &operator<<(ostream &out, info_t info)
{
    for (int i = 0; i < 8; ++i)
        out << info.data.number[i] << " ";
    return out;
}

class test_string_to_info_t
    : public testing::TestWithParam<string_to_info_test_case>
{
};

TEST_P(test_string_to_info_t, test_info_size)
{
    auto cs = GetParam();
    info_t info = stringToInfo(cs.input);
    EXPECT_EQ(info.number_size, cs.number_size);
    EXPECT_EQ(info.text_size, cs.text_length) << info << endl;
}

INSTANTIATE_TEST_SUITE_P(
    stringToInfo,
    test_string_to_info_t,
    testing::Values(
        string_to_info_test_case{"HELLO", 5, 1},
        string_to_info_test_case{"AK07374-XC2-F4{FT1}{FRM-PP}", 27, 4},
        string_to_info_test_case{"AK07374-XC2-F4{FT1}{FRM-PP}111111111", 36, 5},
        string_to_info_test_case{
            "AK07374-XC2-F4{FT1}{FRM-PP}AK07374-XC2-F4{FT1}{FRM-PP}1234567", 61,
            8},
        string_to_info_test_case{
            "AK07374-XC2-F4{FT1}{FRM-PP}AK07374-XC2-F4{FT1}{FRM-PP}123456", 60,
            8},
        string_to_info_test_case{
            "AK07374-XC2-F4{FT1}{FRM-PP}AK07374-XC2-F4{FT1}{FRM-PP}123456789",
            63, 8}

        ));