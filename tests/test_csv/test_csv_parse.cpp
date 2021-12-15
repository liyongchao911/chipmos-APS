#include <gtest/gtest.h>

#define private public
#define protected public

#include <string>
#include <vector>

#include "include/csv.h"

#undef private
#undef protected

struct Text {
    std::vector<std::string> token;

    Text() {}

    void insert(std::string str) { token.push_back(str); }
};

struct TextVector : testing::Test {
    Text *text;

    TextVector() { text = new Text; };

    ~TextVector() { delete text; }
};

struct ParseParameters {
    std::string input;
    char delimiter;
    std::vector<std::string> output;

    friend std::ostream &operator<<(std::ostream &os,
                                    const ParseParameters &obj)
    {
        return os << std::endl << "  >>> " << obj.input << " <<<" << std::endl;
    }
};

struct ParseTest : TextVector, testing::WithParamInterface<ParseParameters> {
    ParseTest() {}
};

TEST_P(ParseTest, csv_parse)
{
    csv_t csv;
    auto as = GetParam();

    char *cpt = strdup(as.input.c_str());

    text->token = csv.parseCsvRow(cpt, as.delimiter);
    EXPECT_EQ(as.output, text->token) << "Expected : as.output\n";

    delete cpt;
}

INSTANTIATE_TEST_SUITE_P(
    Default,
    ParseTest,
    testing::Values(
        // Hello World
        ParseParameters{"hello,world", ',', {"hello", "world"}},

        // Empty String
        ParseParameters{R"("1","2","3",)", ',', {"1", "2", "3", ""}},
        ParseParameters{"", ',', {""}},
        ParseParameters{",,", ',', {"", "", ""}},

        // String Concatenation
        ParseParameters{"str"
                        "1",
                        ',',
                        {"str1"}},

        // Embedded double quote
        ParseParameters{R"("""Quoted""")", ',', {R"("Quoted")"}},

        // Spaces
        ParseParameters{R"(1999, 2000,2001 ,"hello")",
                        ',',
                        {"1999", " 2000", "2001 ", "hello"}},

        // Replace delimiter
        ParseParameters{"hello,world", '^', {"hello,world"}},
        ParseParameters{"hello^world", '^', {"hello", "world"}},
        ParseParameters{R"("1"^"2"^"3"^)", '^', {"1", "2", "3", ""}},
        ParseParameters{"Combine"
                        "String",
                        '^',
                        {"CombineString"}},
        ParseParameters{R"(1997^Ford^E350^"ac, abs, moon"^3000.00)",
                        '^',
                        {"1997", "Ford", "E350", "ac, abs, moon", "3000.00"}},
        ParseParameters{
            R"(1999^Chevy^"Venture ""Extended Edition"""^^5000.00)",
            '^',
            {"1999", "Chevy", "Venture \"Extended Edition\"", "", "5000.00"}},

        // Example (Wikipedia)
        ParseParameters{R"(1997,Ford,E350,"ac, abs, moon",3000.00)",
                        ',',
                        {"1997", "Ford", "E350", "ac, abs, moon", "3000.00"}},
        ParseParameters{
            R"(1999,Chevy,"Venture ""Extended Edition""",,5000.00)",
            ',',
            {"1999", "Chevy", R"(Venture "Extended Edition")", "", "5000.00"}}

        ));
