#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "include/csv.h"
#include "include/infra.h"

#define private public
#define protected public

#include "include/entities.h"

#undef private
#undef protected

using namespace std;

class test_entities_read_dedicate_machine_file_t : public testing::Test
{
protected:
    vector<map<string, bool> > data;


    entities_t *entities;

    void SetUp() override;
    void TearDown() override;
};

void test_entities_read_dedicate_machine_file_t::SetUp()
{
    entities = new entities_t();
    csv_t file("test_dedicate_machine_file.csv", "w");

    map<string, bool> _row;
    map<string, string> output_row;
    bool pass_val;
    for (int i = 0; i < 1000; ++i) {
        string cust = randomDouble() > 0.5 ? "MTI" : "others";
        string entity = "entity" + to_string(i);
        pass_val = randomDouble() > 0.5;
        _row[cust + "_" + entity] = pass_val;

        output_row["customer"] = cust;
        output_row["entity"] = entity;
        output_row["pass"] = pass_val ? "Y" : "N";

        file.addData(output_row);
        data.push_back(_row);
    }
    file.write();
}

void test_entities_read_dedicate_machine_file_t::TearDown()
{
    delete entities;
    remove("test_dedicate_machine_file.csv");
}

TEST_F(test_entities_read_dedicate_machine_file_t,
       test_read_dedicate_machine_file)
{
    entities->_readDedicateMachines("test_dedicate_machine_file.csv");
    // EXPECT_EQ(entities->_dedicate_machines.size(), data.size());
    int size = data.size();
    for (int i = 0; i < size; ++i) {
        for (auto it = data[i].begin(); it != data[i].end(); it++) {
            string key = it->first;
            char *text = strdup(key.c_str());
            vector<string> cust_and_entity = split(text, '_');
            EXPECT_EQ(cust_and_entity.size(), 2);
            string cust, entity;
            cust = cust_and_entity[0];
            entity = cust_and_entity[1];
            EXPECT_EQ(it->second, entities->_dedicate_machines[cust][entity]);
            free(text);
        }
    }
}
