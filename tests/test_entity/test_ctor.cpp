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
    map<string, string> data;

public:
    entity_t *ent;
    void SetUp() override;

    void TearDown() override;
};

void test_entity_t::SetUp()
{
    ent = nullptr;
    data = map<string, string>{{"entity", "BB211"},
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

    ent = new entity_t(data);
    if (ent == nullptr) {
        perror("new entity_t instance");
        exit(EXIT_FAILURE);
    }
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
