#include <gtest/gtest.h>
#include <map>
#include <ostream>
#include <string>


#define private public
#define protected public
#include "include/wip_lot.h"
#undef private
#undef protected

using namespace std;

/**
 * Test if the members are all initialized,
 *  _route == ""
 *  _urgent_code == ""
 *  _last_wb_entity == "";
 *  _sublot_size == 1
 */
class test_wip_lot_default_ctor_t : public testing::Test
{
public:
    lot_wip_t *lot;

    void SetUp() override;
    void TearDown() override;
};

void test_wip_lot_default_ctor_t::SetUp()
{
    lot = new lot_wip_t();
}

void test_wip_lot_default_ctor_t::TearDown()
{
    delete lot;
}


TEST_F(test_wip_lot_default_ctor_t, test_initialize_own_members)
{
    EXPECT_EQ(lot->_route.length(), 0);
    EXPECT_EQ(lot->_urgent_code.length(), 0);
    EXPECT_EQ(lot->_last_wb_entity.length(), 0);
    EXPECT_EQ(lot->_sublot_size, 1);
}


/**
 * The code below test if all lot members has been initialized
 * lot members include :
 *
 *  There are two kinds of TEST suits defined below. One focuses on the row
 * initializer and the other focuses on the default initializer
 */
struct row_initializer_test_case2_t {
    map<string, string> input;
    map<string, string> ans;
    friend ostream &operator<<(std::ostream &os,
                               const row_initializer_test_case2_t &cs)
    {
        return os;
        // os<< endl << "input : "<<endl;
        // for(auto it = input.begin(); it != input.end(); it++){
        //
        // }
    }
};


class test_wip_lot_row_initialized_ctro_t
    : public testing::TestWithParam<row_initializer_test_case2_t>
{
public:
    lot_wip_t *lot_entry;
};

/**
 * In this test suit, test if the row initializer initalizes the member with
 * data in the row and if the data is faulty, the data is given in default value
 * and which will be used to initialize the member
 */
TEST_P(test_wip_lot_row_initialized_ctro_t, test_row_initialized)
{
    struct row_initializer_test_case2_t cs = GetParam();
    ASSERT_NO_THROW(lot_entry = new lot_wip_t(cs.input));
    EXPECT_EQ(lot_entry->_route, cs.ans["route"]);
    EXPECT_EQ(lot_entry->_urgent_code, cs.ans["urgent_code"]);
    EXPECT_EQ(lot_entry->_last_wb_entity, cs.ans["last_wb_entity"]);
    EXPECT_EQ(lot_entry->_sublot_size, stoi(cs.ans["sub_lot_size"]));
    delete lot_entry;
}


INSTANTIATE_TEST_SUITE_P(
    rowInitializer2,
    test_wip_lot_row_initialized_ctro_t,
    testing::Values(row_initializer_test_case2_t{{{"lot_number", "PXX12345"},
                                                  {"pin_package", "pinpackage"},
                                                  {"recipe", "BBDFRECIPE"},
                                                  {"prod_id", "PROD_ID"},
                                                  {"part_no", "PART_NO"},
                                                  {"pkg_id", "PKG_ID"},
                                                  {"customer", "CUSTOMER"},
                                                  {"oper", "2020"},
                                                  {"qty", "1999"},
                                                  {"route", "ROUTE"},
                                                  {"urgent_code", "A"},
                                                  {"last_wb_entity", "BB356"},
                                                  {"sub_lot_size", "20"}},
                                                 {{"lot_number", "PXX12345"},
                                                  {"pin_package", "pinpackage"},
                                                  {"recipe", "BBDFRECIPE"},
                                                  {"prod_id", "PROD_ID"},
                                                  {"part_no", "PART_NO"},
                                                  {"pkg_id", "PKG_ID"},
                                                  {"customer", "CUSTOMER"},
                                                  {"oper", "2020"},
                                                  {"qty", "1999"},
                                                  {"route", "ROUTE"},
                                                  {"urgent_code", "A"},
                                                  {"last_wb_entity", "BB356"},
                                                  {"sub_lot_size", "20"}}},
                    row_initializer_test_case2_t{{{"lot_number", "PXX12345"},
                                                  {"pin_package", "pinpackage"},
                                                  {"recipe", "BBDFRECIPE"},
                                                  {"prod_id", "PROD_ID"},
                                                  {"part_no", "PART_NO"},
                                                  {"pkg_id", "PKG_ID"},
                                                  {"customer", "CUSTOMER"},
                                                  {"route", "ROUTE"},
                                                  {"last_wb_entity", "BB356"}},
                                                 {{"lot_number", "PXX12345"},
                                                  {"pin_package", "pinpackage"},
                                                  {"recipe", "BBDFRECIPE"},
                                                  {"prod_id", "PROD_ID"},
                                                  {"part_no", "PART_NO"},
                                                  {"pkg_id", "PKG_ID"},
                                                  {"customer", "CUSTOMER"},
                                                  {"oper", "0"},
                                                  {"qty", "0"},
                                                  {"route", "ROUTE"},
                                                  {"urgent_code", ""},
                                                  {"last_wb_entity", "BB356"},
                                                  {"sub_lot_size", "1"}}},
                    row_initializer_test_case2_t{{},
                                                 {{"lot_number", ""},
                                                  {"pin_package", ""},
                                                  {"recipe", ""},
                                                  {"prod_id", ""},
                                                  {"part_no", ""},
                                                  {"pkg_id", ""},
                                                  {"customer", ""},
                                                  {"oper", "0"},
                                                  {"qty", "0"},
                                                  {"route", ""},
                                                  {"urgent_code", ""},
                                                  {"last_wb_entity", ""},
                                                  {"sub_lot_size", "1"}}}));
