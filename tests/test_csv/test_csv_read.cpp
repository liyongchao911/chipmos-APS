#include <gtest/gtest.h>
#include "include/def.h"

#define private public
#define protected public

#include "include/csv.h"

#undef private
#undef protected

#include <map>
#include <string>
#include <vector>

using namespace std;

struct test_data_t {
    string filename;
    vector<string> headers;
    vector<vector<string> > content;
};

class test_csv_read_t : public testing::Test
{
public:
    bool twoVectorsAreTheSame(vector<string> t1, vector<string> t2);
    string insertBom(string text, unsigned int bom);
};

bool test_csv_read_t::twoVectorsAreTheSame(vector<string> t1, vector<string> t2)
{
    if (t1.size() != t2.size())
        return false;

    int size = t1.size();
    for (int i = 0; i < size; ++i) {
        if (t1[i].compare(t2[i]) != 0) {
            return false;
        }
    }
    return true;
}

string test_csv_read_t::insertBom(string text, unsigned int bom)
{
    typedef union bom_union {
        unsigned int bom;
        char text[4];
    } bom_union_t;

    bom_union_t _bom;
    _bom.bom = bom;
    text = _bom.text + text;
    return text;
}

TEST_F(test_csv_read_t, test_ctor_1)
{
    csv_t csv;
    EXPECT_EQ(csv._file, nullptr);
    EXPECT_EQ(csv._data.size(), 0);
    EXPECT_EQ(csv._head.size(), 0);
    EXPECT_EQ(csv._mode.length(), 0);
    EXPECT_EQ(csv._filename.length(), 0);
}

TEST_F(test_csv_read_t, test_ctor_with_2_params)
{
    csv_t csv("filename", "r");
    EXPECT_EQ(csv._filename.compare("filename"), 0);
    EXPECT_EQ(csv._mode.compare("r"), 0);
    EXPECT_EQ(csv._file, nullptr);
}

TEST_F(test_csv_read_t, test_set_mode)
{
    csv_t csv("filename", "r");
    EXPECT_EQ(csv._mode.compare("r"), 0);
    csv.setMode("w");
    EXPECT_EQ(csv._mode.compare("w"), 0);
}

TEST_F(test_csv_read_t, test_set_filename)
{
    csv_t csv("filename", "r");
    EXPECT_EQ(csv._filename.compare("filename"), 0);
    csv.setFileName("filename2");
    EXPECT_EQ(csv._filename.compare("filename2"), 0);
}

TEST_F(test_csv_read_t, test_parse_csv_row)
{
    csv_t csv;
    const char *test_data = "1,2,3,";
    vector<string> ans = {"1", "2", "3", ""};

    typedef struct test_parse_data_t {
        string text;
        vector<string> ans;
        char delimiter;
    } test_parse_data_t;

    vector<test_parse_data_t> data;
    data.push_back(test_parse_data_t{.text = "1,2,3,",
                                     .ans = vector<string>({"1", "2", "3", ""}),
                                     .delimiter = ','});

    data.push_back(
        test_parse_data_t{.text = "1|2|3||",
                          .ans = vector<string>({"1", "2", "3", "", ""}),
                          .delimiter = '|'});
    // TODO : write down some testing data

    for (unsigned int i = 0; i < data.size(); ++i) {
        char *dup_data = strdup(data[i].text.c_str());
        vector<string> out = csv.parseCsvRow(dup_data, data[i].delimiter);
        EXPECT_TRUE(twoVectorsAreTheSame(out, data[i].ans));
        free(dup_data);
    }
}

TEST_F(test_csv_read_t, test_has_bom)
{
    typedef struct test_has_bom_data_t {
        unsigned int bom;
        string text;
        bool insert_bom;
        bool ans;
    } test_has_bom_data_t;

    vector<test_has_bom_data_t> data;
    csv_t csv;

    // UTF-8
    data.push_back(test_has_bom_data_t{
        .bom = 0x00BFBBEF, .text = "TEXT", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x00BFBBEF, .text = "", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x00BFBBEF, .text = "TEXT", .insert_bom = false, .ans = false});

    // UTF-16
    data.push_back(test_has_bom_data_t{
        .bom = 0x0000FFFE, .text = "TEXT", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x0000FFFE, .text = "", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x0000FFFE, .text = "TEXT", .insert_bom = false, .ans = false});

    // UTF-7
    data.push_back(test_has_bom_data_t{
        .bom = 0x00762F2B, .text = "TEXT", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x00762F2B, .text = "", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x00762F2B, .text = "TEXT", .insert_bom = false, .ans = false});

    // UTF-1
    data.push_back(test_has_bom_data_t{
        .bom = 0x004C64F7, .text = "TEXT", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x004C64F7, .text = "", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x004C64F7, .text = "TEXT", .insert_bom = false, .ans = false});

    // EDBIC
    data.push_back(test_has_bom_data_t{
        .bom = 0x736673DD, .text = "TEXT", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x736673DD, .text = "", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x736673DD, .text = "TEXT", .insert_bom = false, .ans = false});

    // SCSU
    data.push_back(test_has_bom_data_t{
        .bom = 0x00FFFE0E, .text = "TEXT", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x00FFFE0E, .text = "", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x00FFFE0E, .text = "TEXT", .insert_bom = false, .ans = false});

    // BOCU-1
    data.push_back(test_has_bom_data_t{
        .bom = 0x0028EEFB, .text = "TEXT", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x0028EEFB, .text = "", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x0028EEFB, .text = "TEXT", .insert_bom = false, .ans = false});

    // GB-18030
    data.push_back(test_has_bom_data_t{
        .bom = 0x33953184, .text = "TEXT", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x33953184, .text = "", .insert_bom = true, .ans = true});

    data.push_back(test_has_bom_data_t{
        .bom = 0x33953184, .text = "TEXT", .insert_bom = false, .ans = false});

    // TODO : Please write down some testing data below

    for (unsigned int i = 0; i < data.size(); ++i) {
        string test_text = data[i].text;
        if (data[i].insert_bom) {
            test_text = insertBom(test_text, data[i].bom);
        }
        char *text = strdup(test_text.c_str());
        bool retval =
            csv._hasBOM(text, data[i].bom, (short) __builtin_clz(data[i].bom));
        EXPECT_EQ(retval, data[i].ans);
        free(text);
    }
}

TEST_F(test_csv_read_t, test_csv_read_exception_handling)
{
    csv_t *csv;
    csv = new csv_t("filename", "r");
    EXPECT_THROW(csv->read(), ios_base::failure);
    delete csv;

    csv = new csv_t("filename", "P");
    EXPECT_THROW(csv->read(), invalid_argument);
    delete csv;
}

TEST_F(test_csv_read_t, test_csv_read_file_header)
{
    typedef struct test_csv_read_data_t {
        string header;
        vector<string> content;
        map<string, int> ans;
    } test_csv_read_data_t;

    vector<test_csv_read_data_t> data;
    data.push_back(test_csv_read_data_t{
        .header = "h1,h2,h3", .ans = {{"h1", 0}, {"h2", 1}, {"h3", 2}}});


    data.push_back(test_csv_read_data_t{
        .header = "h1,h2,h3,h4,h5",
        .ans = {{"h1", 0}, {"h2", 1}, {"h3", 2}, {"h4", 3}, {"h5", 4}}});

    data.push_back(test_csv_read_data_t{.header = ""});

    const char *filename = "testing_file.csv";
    for (unsigned int i = 0; i < data.size(); ++i) {
        FILE *file = fopen(filename, "w");
        string header = data[i].header;
        vector<string> content = data[i].content;
        map<string, int> ans = data[i].ans;

        if (header.length()) {
            fprintf(file, "%s\n", (char *) header.c_str());
        }

        for (unsigned int j = 0; j < content.size(); ++j) {
            fprintf(file, "%s\n", (char *) content[j].c_str());
        }
        fclose(file);

        csv_t *csv = new csv_t(filename, "r");
        EXPECT_NO_THROW(csv->read());
        for (auto it = ans.begin(); it != ans.end(); ++it) {
            EXPECT_NO_THROW(csv->_head.at(it->first));
            EXPECT_EQ(csv->_head.at(it->first), it->second);
        }
    }
    remove(filename);
}
