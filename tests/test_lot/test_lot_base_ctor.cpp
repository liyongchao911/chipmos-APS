#include <gtest/gtest.h>
#include <map>
#include <string>

using namespace std;

#define private public
#define protected public
#include "include/lot_base.h"
#undef private
#undef protected

/**
 * Test if all members, including the member from inherited class have been
 * initialized.
 */
class test_lot_base_default_ctor_t : public testing::Test
{
public:
    lot_base_t *lot;

    void SetUp() override;
    void TearDown() override;
};

void test_lot_base_default_ctor_t::SetUp()
{
    lot = new lot_base_t();
}

void test_lot_base_default_ctor_t::TearDown()
{
    delete lot;
}

/**
 * In this test fixture, test if all inherited members are initialized
 */
TEST_F(test_lot_base_default_ctor_t, test_initialize_inherited_members)
{
    EXPECT_EQ(lot->next, nullptr) << "lot->next is not initialized to nullptr";
    EXPECT_EQ(lot->prev, nullptr) << "lot->prev is not initialized to nullptr";
    EXPECT_EQ(lot->get_value, nullptr)
        << "lot->get_value is not initialized to nullptr";
    EXPECT_EQ(lot->list_ele_t::ptr_derived_object, nullptr)
        << "list_ele_t::ptr_derived_object is not initialized";
    EXPECT_EQ(lot->job_base_t::ptr_derived_object, nullptr)
        << "job_base_t::ptr_derived_object is not initialized";
    EXPECT_EQ(lot->ms_gene, nullptr) << "lot->ms_gene is not initialized";
    EXPECT_EQ(lot->os_seq_gene, nullptr)
        << "lot->os_seq_gene is not initialized";
    EXPECT_EQ(lot->partition, 0.0) << "lot->partition is not initialized";
    EXPECT_EQ(lot->process_time, nullptr)
        << "lot->process_time is not initialized";
    EXPECT_EQ(lot->size_of_process_time, 0)
        << "lot->size_of_process_time is not initialized";
    EXPECT_EQ(lot->qty, 0) << "lot->qty is not initialzied";
    EXPECT_EQ(lot->current_machine, nullptr)
        << "lot->current_machine is not initialzied";
    EXPECT_EQ(lot->arriv_t, 0.0) << "lot->arriv_t is not initialized";
    EXPECT_EQ(lot->start_time, 0.0) << "lot->start_time is not initialized";
    EXPECT_EQ(lot->end_time, 0.0) << "lot->end_time is not initialzied";
    EXPECT_EQ(lot->ptime, 0.0) << "lot->ptime is not initialized";
}


TEST_F(test_lot_base_default_ctor_t, test_initialize_own_members)
{
    EXPECT_EQ(lot->_lot_number.length(), 0);
    EXPECT_EQ(lot->_pin_package.length(), 0);
    EXPECT_EQ(lot->_recipe.length(), 0);
    EXPECT_EQ(lot->_prod_id.length(), 0);
    EXPECT_EQ(lot->_part_no.length(), 0);
    EXPECT_EQ(lot->_pkg_id.length(), 0);
    EXPECT_EQ(lot->_customer.length(), 0);

    EXPECT_EQ(lot->_oper, 0);
    EXPECT_EQ(lot->_qty, 0);
    EXPECT_EQ(lot->_number_of_wires, 0);
    EXPECT_EQ(lot->_number_of_tools, 0);
    EXPECT_FALSE(lot->_hold);
    EXPECT_FALSE(lot->_mvin);
    EXPECT_FALSE(lot->_is_sub_lot);
    EXPECT_FALSE(lot->_spr_hot);
    EXPECT_EQ(lot->_cr, 0.0);
}


/**
 * The code below test if all lot members has been initialized
 * lot members include :
 *  _lot_number,
 *  _pin_package,
 *  _recipe,
 *  _prod_id,
 *  _part_id,
 *  _pkg_id,
 *  _customer,
 *
 *  _qty,
 *  _oper,
 *  _number_of_wires,
 *  _number_of_tools,
 *
 *  _hold,
 *  _mvin,
 *  _is_sub_lot;
 *  _spr_hot
 *  _cr
 *
 *  There are two kinds of TEST suits defined below. One focuses on the row
 * initializer and the other focuses on the default initializer
 */

struct row_initializer_test_case_t {
    map<string, string> input;
    map<string, string> ans;
};

class test_lot_base_row_initialized_ctro_t
    : public testing::TestWithParam<struct row_initializer_test_case_t>
{
public:
    lot_base_t *lot_base;
};

/**
 * In this test suit, test if the row initializer initalizes the member with
 * data in the row and if the data is faulty, the data is given in default value
 * and which will be used to initialize the member
 */
TEST_P(test_lot_base_row_initialized_ctro_t, test_lot_base_row_initialized)
{
    auto cs = GetParam();
    lot_base = new lot_base_t(cs.input);
    EXPECT_EQ(lot_base->_lot_number, cs.ans["lot_number"]);
    EXPECT_EQ(lot_base->_pin_package, cs.ans["pin_package"]);
    EXPECT_EQ(lot_base->_recipe, cs.ans["recipe"]);
    EXPECT_EQ(lot_base->_prod_id, cs.ans["prod_id"]);
    EXPECT_EQ(lot_base->_part_no, cs.ans["part_no"]);
    EXPECT_EQ(lot_base->_pkg_id, cs.ans["pkg_id"]);
    EXPECT_EQ(lot_base->_customer, cs.ans["customer"]);

    EXPECT_EQ(lot_base->_oper, stoi(cs.ans["oper"]));
    EXPECT_EQ(lot_base->_qty, stoi(cs.ans["qty"]));
    EXPECT_EQ(lot_base->_number_of_wires, 0);
    EXPECT_EQ(lot_base->_number_of_tools, 0);
    EXPECT_FALSE(lot_base->_hold);
    EXPECT_FALSE(lot_base->_mvin);
    EXPECT_FALSE(lot_base->_is_sub_lot);
    EXPECT_FALSE(lot_base->_spr_hot);
    EXPECT_EQ(lot_base->_cr, 0.0);
    delete lot_base;
}


INSTANTIATE_TEST_SUITE_P(
    rowInitializer,
    test_lot_base_row_initialized_ctro_t,
    testing::Values(row_initializer_test_case_t{{{"lot_number", "PXX12345"},
                                                 {"pin_package", "pinpackage"},
                                                 {"recipe", "BBDFRECIPE"},
                                                 {"prod_id", "PROD_ID"},
                                                 {"part_no", "PART_NO"},
                                                 {"pkg_id", "PKG_ID"},
                                                 {"customer", "CUSTOMER"},
                                                 {"oper", "2020"},
                                                 {"qty", "1999"}},
                                                {{"lot_number", "PXX12345"},
                                                 {"pin_package", "pinpackage"},
                                                 {"recipe", "BBDFRECIPE"},
                                                 {"prod_id", "PROD_ID"},
                                                 {"part_no", "PART_NO"},
                                                 {"pkg_id", "PKG_ID"},
                                                 {"customer", "CUSTOMER"},
                                                 {"oper", "2020"},
                                                 {"qty", "1999"}}},
                    row_initializer_test_case_t{
                        {
                            {"lot_number", "PXX12345"},
                            {"pin_package", "pinpackage"},
                            {"recipe", "BBDFRECIPE"},
                            {"prod_id", "PROD_ID"},
                            {"part_no", "PART_NO"},
                            {"pkg_id", "PKG_ID"},
                            {"customer", "CUSTOMER"},
                        },
                        {{"lot_number", "PXX12345"},
                         {"pin_package", "pinpackage"},
                         {"recipe", "BBDFRECIPE"},
                         {"prod_id", "PROD_ID"},
                         {"part_no", "PART_NO"},
                         {"pkg_id", "PKG_ID"},
                         {"customer", "CUSTOMER"},
                         {"oper", "0"},
                         {"qty", "0"}}},
                    row_initializer_test_case_t{{},
                                                {{"lot_number", ""},
                                                 {"pin_package", ""},
                                                 {"recipe", ""},
                                                 {"prod_id", ""},
                                                 {"part_no", ""},
                                                 {"pkg_id", ""},
                                                 {"customer", ""},
                                                 {"oper", "0"},
                                                 {"qty", "0"}}}));
