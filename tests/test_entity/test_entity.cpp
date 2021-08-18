#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>


#include <gtest/gtest.h>

#define private public
#define protected public

#include "include/entity.h"
#include "include/lot.h"

#undef private
#undef protected

using namespace std;

class test_entity_t : public testing::Test
{
private:
    map<string, string> entity_data;
    map<string, string> lot_data_template;

public:
    map<string, string> lot_data_case1;
    map<string, string> lot_data_case2;
    map<string, string> lot_data_case3;

    entity_t *ent;

    lot_t *createALot(map<string, string> data);

    void SetUp() override;

    void TearDown() override;
};


lot_t *test_entity_t::createALot(map<string, string> data)
{
    lot_t *lot = nullptr;
    lot = new lot_t(data);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    return lot;
}

void test_entity_t::SetUp()
{
    ent = nullptr;
    entity_data = map<string, string>{{"entity", "BB211"},
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
                                      {"part_no", "PART_NO"},
                                      {"part_id", "PART_ID"}};

    ent = new entity_t(entity_data);
    if (ent == nullptr) {
        perror("new entity_t instance");
        exit(EXIT_FAILURE);
    }


    lot_data_template = map<string, string>({{"route", "QFNS288"},
                                             {"lot_number", "P23ASEA02"},
                                             {"pin_package", "DFN-08YM2G"},
                                             {"bd_id", "AAS008YM2024A"},
                                             {"prod_id", "008YMAS034"},
                                             {"urgent_code", "urgent"},
                                             {"customer", "MXIC"},
                                             {"wb_location", "BB211-1"},
                                             {"qty", "16000"},
                                             {"oper", "2200"},
                                             {"hold", "N"},
                                             {"mvin", "Y"},
                                             {"sub_lot", "9"},
                                             {"part_id", "PART_ID"},
                                             {"part_no", "PART_NO"}});

    lot_data_case1 = lot_data_template;

    lot_data_case2 = lot_data_template;
    lot_data_case2["wb_location"] = "NB234";


    lot_data_case3 = lot_data_template;
    lot_data_case3["wb_location"] = "BB211-2";
}


void test_entity_t::TearDown()
{
    delete ent;
}

TEST_F(test_entity_t, test_entity_setup)
{
    EXPECT_EQ(ent->_entity_name.compare("BB211"), 0);
    EXPECT_EQ(ent->_model_name.compare("UTC3000"), 0);
    EXPECT_EQ(ent->_location.compare("TA-A"), 0);
}

TEST_F(test_entity_t, test_entity_current_lot_setup)
{
    EXPECT_EQ(ent->_current_lot._lot_number.compare("P23AWDV31"), 0);
    EXPECT_EQ(ent->_current_lot._pin_package.compare("TSOP1-48/M2"), 0);
    EXPECT_EQ(ent->_current_lot._recipe.compare("AAW048TP1041B"), 0);
    EXPECT_EQ(ent->_current_lot._customer.compare("MXIC"), 0);
    EXPECT_EQ(ent->_current_lot._qty, 1280);
    EXPECT_EQ(ent->_current_lot._oper, 2200);
    EXPECT_EQ(ent->_current_lot._part_id.compare("PART_ID"), 0);
    EXPECT_EQ(ent->_current_lot._part_no.compare("PART_NO"), 0);
}

TEST_F(test_entity_t, test_entity_addPrescheduledLot_successful)
{
    lot_t *lot = createALot(lot_data_case1);

    ent->addPrescheduledLot(lot);
    EXPECT_EQ(ent->_prescheduled_lots[0], lot);

    delete lot;
    lot = nullptr;
}

TEST_F(test_entity_t, test_entity_addPrescheduledLot_failed)
{
    lot_t *lot = createALot(lot_data_case2);

    EXPECT_THROW(ent->addPrescheduledLot(lot), invalid_argument);

    delete lot;
    lot = nullptr;
}

TEST_F(test_entity_t, test_entity_prescheduleLots1)
{
    lot_t *lot1 = createALot(lot_data_case1);
    lot_t *lot2 = createALot(lot_data_case3);

    ent->addPrescheduledLot(lot1);
    ent->addPrescheduledLot(lot2);

    ent->prescheduleLots();

    EXPECT_EQ(ent->_prescheduled_lots[0], lot1);
    EXPECT_EQ(ent->_prescheduled_lots[1], lot2);


    delete lot1;
    delete lot2;
}

TEST_F(test_entity_t, test_entity_prescheduleLots2)
{
    lot_t *lot1 = createALot(lot_data_case1);
    lot_t *lot2 = createALot(lot_data_case3);

    ent->addPrescheduledLot(lot2);
    ent->addPrescheduledLot(lot1);

    ent->prescheduleLots();

    EXPECT_EQ(ent->_prescheduled_lots[0], lot1);
    EXPECT_EQ(ent->_prescheduled_lots[1], lot2);


    delete lot1;
    delete lot2;
}
