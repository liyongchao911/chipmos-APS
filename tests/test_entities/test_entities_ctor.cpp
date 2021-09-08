#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#define private public
#define protected public

#include "include/entities.h"
#include "include/entity.h"

#undef private
#undef protected

using namespace std;

class test_entities_t : public testing::Test
{
private:
    map<string, string> data;
    map<string, string> prod_map_to_pid;
    map<string, string> prod_map_to_bomid;
    map<string, string> pid_map_to_part_no;
    map<string, string> bomid_map_to_part_id;

public:
    entities_t *entities;

    map<string, string> getData() { return data; }
    void SetUp() override;
    void TearDown() override;
};

void test_entities_t::SetUp()
{
    entities = nullptr;
    data = map<string, string>{
        {"entity", "BB211"},
        {"model", "UTC3000"},
        {"recover_time", "2021/6/19 15:24"},
        {"prod_id", "048TPAW086"},
        {"pin_package", "TSOP1-48/M2"},
        {"lot_number", "P23AWDV31"},
        {"customer", "MXIC"},
        {"bd_id", "AAW048TP1041B"},
        {"oper", "2200"},
        {"qty", "1280"},
        {"location", "TA-A"},
    };

    prod_map_to_pid = map<string, string>({
        {"048TPAW086", "PID"},
    });

    prod_map_to_bomid = map<string, string>({
        {"048TPAW086", "BOMID"},
    });

    pid_map_to_part_no = map<string, string>({
        {"PID", "PART_NO"},
    });

    bomid_map_to_part_id = map<string, string>({{"BOMID_2200", "PART_ID"}});

    entities = new entities_t();

    if (entities == nullptr) {
        exit(EXIT_FAILURE);
    } else {
        entities->prod_map_to_bom_id = prod_map_to_bomid;
        entities->prod_map_to_pid = prod_map_to_pid;

        entities->pid_map_to_part_no = pid_map_to_part_no;
        entities->bom_id_map_to_part_id = bomid_map_to_part_id;
    }
}

void test_entities_t::TearDown()
{
    if (entities == nullptr)
        delete entities;
    entities = nullptr;
}

TEST_F(test_entities_t, test_entities_setup)
{
    EXPECT_EQ(entities->_time, 0);
}

TEST_F(test_entities_t, test_addMachine_invalid_recover_time)
{
    map<string, string> data = getData();
    data.erase(data.find("recover_time"));
    EXPECT_THROW(entities->addMachine(data), invalid_argument);
}

TEST_F(test_entities_t, test_addMachine_entity_current_lot_setup)
{
    map<string, string> data = getData();
    entity_t *ent = entities->addMachine(data);

    EXPECT_EQ(ent->_current_lot->_lot_number.compare("P23AWDV31"), 0);
    EXPECT_EQ(ent->_current_lot->_pin_package.compare("TSOP1-48/M2"), 0);
    EXPECT_EQ(ent->_current_lot->_recipe.compare("AAW048TP1041B"), 0);
    EXPECT_EQ(ent->_current_lot->_customer.compare("MXIC"), 0);
    EXPECT_EQ(ent->_current_lot->_qty, 1280);
    EXPECT_EQ(ent->_current_lot->_oper, 2200);
    EXPECT_EQ(ent->_current_lot->_part_id.compare("PART_ID"), 0);
    EXPECT_EQ(ent->_current_lot->_part_no.compare("PART_NO"), 0);
}

TEST_F(test_entities_t, test_addMachine_entities_containers)
{
    map<string, string> data = getData();
    entity_t *ent = entities->addMachine(data);
    EXPECT_EQ(entities->_ents.back(), ent);
    EXPECT_NO_THROW(entities->_entities.at(ent->_model_name));
    EXPECT_NO_THROW(
        entities->_entities.at(ent->_model_name).at(ent->_location));
    EXPECT_EQ(entities->_entities[ent->_model_name][ent->_location].back(),
              ent);
    EXPECT_EQ(entities->_model_locations[ent->_model_name].back().compare(
                  ent->_location),
              0);
    EXPECT_NO_THROW(entities->name_entity.at(ent->_entity_name));
    EXPECT_EQ(entities->name_entity[ent->_entity_name], ent);
}

TEST_F(test_entities_t, test_setTime)
{
    string text1("2021/6/19 15:24");
    string text2("");
    entities->setTime(text1);
    EXPECT_NE(entities->_time, 0);

    entities->setTime(text2);
    EXPECT_EQ(entities->_time, 0);
}

TEST_F(test_entities_t, test_getEntityByName)
{
    map<string, string> data = getData();
    entity_t *ent = entities->addMachine(data);

    entity_t *case1 = entities->getEntityByName("BB666");
    EXPECT_EQ(case1, nullptr);

    entity_t *case2 = entities->getEntityByName("BB211");
    EXPECT_EQ(case2, ent);
}
