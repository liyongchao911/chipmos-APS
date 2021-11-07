#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#define private public
#define protected public

#include "include/lots.h"

#undef private
#undef protected


using namespace std;


class test_lots_t : public testing::Test
{
private:
    map<string, string> _testing_data_template;

public:
    map<string, string> case1;
    map<string, string> case2;

    lot_t *lot;
    lots_t *lots;

    void SetUp() override;

    void TearDown() override;
};

void test_lots_t::TearDown()
{
    if (lots)
        delete lots;
    lots = nullptr;
}

void test_lots_t::SetUp()
{
    lot = nullptr;

    lots = nullptr;
    lots = new lots_t();
    if (lots == nullptr)
        exit(EXIT_FAILURE);


    _testing_data_template = map<string, string>({{"route", "QFNS288"},
                                                  {"lot_number", "P23ASEA02"},
                                                  {"pin_package", "DFN-08YM2G"},
                                                  {"bd_id", "AAS008YM2024A"},
                                                  {"prod_id", "008YMAS034"},
                                                  {"urgent_code", ""},
                                                  {"customer", ""},
                                                  {"wb_location", ""},
                                                  {"qty", "16000"},
                                                  {"oper", "2200"},
                                                  {"hold", "N"},
                                                  {"mvin", "Y"},
                                                  {"sub_lot", "9"},
                                                  {"queue_time", ""},
                                                  {"fcst_time", ""},
                                                  {"amount_of_tools", ""},
                                                  {"amount_of_wires", ""},
                                                  {"CAN_RUN_MODELS", ""},
                                                  {"PROCESS_TIME", ""},
                                                  {"uphs", ""},
                                                  {"part_id", ""},
                                                  {"part_no", ""}});

    case1 = map<string, string>({{"route", "QFNS288"},
                                 {"lot_number", "P23ASEA02"},
                                 {"pin_package", "DFN-08YM2G"},
                                 {"bd_id", "AAS008YM2024A"},
                                 {"prod_id", "008YMAS034"},
                                 {"urgent_code", "urgent"},
                                 {"customer", "MXIC"},
                                 {"wb_location", "NB123-1"},
                                 {"qty", "16000"},
                                 {"oper", "2200"},
                                 {"hold", "N"},
                                 {"mvin", "Y"},
                                 {"sub_lot", "9"},
                                 {"amount_of_tools", "100"},
                                 {"amount_of_wires", "20"},
                                 {"CAN_RUN_MODELS", ""},
                                 {"PROCESS_TIME", ""},
                                 {"uphs", ""},
                                 {"part_id", "PART_ID"},
                                 {"part_no", "PART_NO"}});

    case2 = map<string, string>({{"route", "QFNS288"},
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
                                 {"amount_of_tools", "100"},
                                 {"amount_of_wires", "20"},
                                 {"CAN_RUN_MODELS", ""},
                                 {"PROCESS_TIME", ""},
                                 {"uphs", ""},
                                 {"part_id", "PART_ID"},
                                 {"part_no", "PART_NO"}});
}

TEST_F(test_lots_t, test_lots_case1)
{
    lot = new lot_t(case1);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    vector<lot_t *> _all_lots;
    _all_lots.push_back(lot);

    lots->addLots(_all_lots);

    EXPECT_EQ(lots->prescheduled_lots.size(), 0);
    EXPECT_EQ(lots->lots.size(), 1);

    EXPECT_NO_THROW(lots->tool_lots.at(lot->part_no()));
    EXPECT_EQ(lots->tool_lots.at(lot->part_no())[0], lot);

    EXPECT_NO_THROW(lots->wire_lots.at(lot->_part_id));
    EXPECT_EQ(lots->wire_lots.at(lot->_part_id)[0], lot);

    EXPECT_NO_THROW(
        lots->tool_wire_lots.at(lot->part_no() + "_" + lot->_part_id));
    EXPECT_EQ(lots->tool_wire_lots.at(lot->part_no() + "_" + lot->_part_id)[0],
              lot);


    EXPECT_EQ(lots->amount_of_tools.at(lot->part_no()),
              lot->getAmountOfTools());
    EXPECT_EQ(lots->amount_of_wires.at(lot->_part_id), lot->_amount_of_wires);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lots_t, test_lots_case2)
{
    lot = new lot_t(case2);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    vector<lot_t *> _all_lots;
    _all_lots.push_back(lot);

    lots->addLots(_all_lots);

    EXPECT_EQ(lots->prescheduled_lots.size(), 1);
    EXPECT_EQ(lots->lots.size(), 0);

    EXPECT_ANY_THROW(lots->tool_lots.at(lot->part_no()));
    EXPECT_ANY_THROW(lots->wire_lots.at(lot->_part_id));
    EXPECT_ANY_THROW(
        lots->tool_wire_lots.at(lot->part_no() + "_" + lot->_part_id));


    delete lot;
    lot = nullptr;
}
