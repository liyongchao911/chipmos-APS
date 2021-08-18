#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#define private public
#define protected public

#include "include/lot.h"

#undef private
#undef protected

using namespace std;

class test_lot_t : public testing::Test
{
private:
    map<string, string> _template;

public:
    // for the information from wip
    map<string, string> test_wip_data_1;
    map<string, string> test_wip_data_2;
    map<string, string> test_wip_data_3;
    map<string, string> test_wip_data_4;
    map<string, string> test_wip_data_5;

    map<string, string> test_lot_csv_data_1;
    map<string, string> test_lot_csv_data_2;
    map<string, string> test_lot_csv_data_3;


    lot_t *lot;

    void SetUp() override;
};

void test_lot_t::SetUp()
{
    lot = nullptr;
    _template = map<string, string>({{"route", "QFNS288"},
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


    test_wip_data_1 = map<string, string>({
        {"route", "QFNS288"},
        {"lot_number", "P23ASEA02"},
        {"pin_package", "DFN-08YM2G"},
        {"bd_id", "AAS008YM2024A"},
        {"prod_id", "008YMAS034"},
        {"urgent_code", "urgent"},
        {"customer", "ICSI"},
        {"wb_location", "BB211-1"},
        {"qty", "16000"},
        {"oper", "2200"},
        {"hold", "N"},
        {"mvin", "Y"},
        {"sub_lot", "9"},
    });

    test_wip_data_2 = map<string, string>({
        {"route", "QFNS288"},
        {"lot_number", "P23ASEA"},
        {"pin_package", "DFN-08YM2G"},
        {"bd_id", "AAS008YM2024A"},
        {"prod_id", "008YMAS034"},
        {"urgent_code", "urgent"},
        {"customer", "ICSI"},
        {"wb_location", "BB211-1"},
        {"qty", ""},
        {"oper", "2200"},
        {"hold", "N"},
        {"mvin", "N"},
        {"sub_lot", "9"},
    });

    test_wip_data_3 = map<string, string>({
        {"route", "QFNS288"},
        {"lot_number", "P23ASEA"},
        {"pin_package", "DFN-08YM2G"},
        {"bd_id", "AAS008YM2024A"},
        {"prod_id", "008YMAS034"},
        {"urgent_code", "urgent"},
        {"customer", "ICSI"},
        {"wb_location", "BB211"},
        {"qty", ""},
        {"oper", "2200"},
        {"hold", "N"},
        {"mvin", "N"},
        {"sub_lot", "9"},
    });

    test_wip_data_4 = map<string, string>({
        {"route", "QFNS288"},
        {"lot_number", "P23ASEA"},
        {"pin_package", "DFN-08YM2G"},
        {"bd_id", "AAS008YM2024A"},
        {"prod_id", "008YMAS034"},
        {"urgent_code", "urgent"},
        {"customer", "ICSI"},
        {"wb_location", "NB124"},
        {"qty", ""},
        {"oper", "2200"},
        {"hold", "N"},
        {"mvin", "N"},
        {"sub_lot", "9"},
    });

    test_wip_data_4 = map<string, string>({
        {"route", "QFNS288"},
        {"lot_number", "P23ASEA"},
        {"pin_package", "DFN-08YM2G"},
        {"bd_id", "AAS008YM2024A"},
        {"prod_id", "008YMAS034"},
        {"urgent_code", "urgent"},
        {"customer", "ICSI"},
        {"wb_location", ""},
        {"qty", ""},
        {"oper", "2200"},
        {"hold", "N"},
        {"mvin", "N"},
        {"sub_lot", "9"},
    });



    test_lot_csv_data_1 =
        map<string, string>({{"route", "QFNS288"},
                             {"lot_number", "P23ASEA02"},
                             {"pin_package", "DFN-08YM2G"},
                             {"bd_id", "AAS008YM2024A"},
                             {"prod_id", "008YMAS034"},
                             {"urgent_code", "urgent"},
                             {"customer", ""},
                             {"wb_location", ""},
                             {"qty", "16000"},
                             {"oper", "2200"},
                             {"hold", "N"},
                             {"mvin", "Y"},
                             {"sub_lot", "9"},
                             {"queue_time", "240.345"},
                             {"fcst_time", "123.2"},
                             {"amount_of_tools", "10"},
                             {"amount_of_wires", "20"},
                             {"CAN_RUN_MODELS", "UTC1000S,UTC2000S,UTC3000S"},
                             {"PROCESS_TIME", "123.45,456.78,789.1"},
                             {"uphs", "23,45,67"},
                             {"part_id", "PART_ID"},
                             {"part_no", "PART_NO"}});

    test_lot_csv_data_2 =
        map<string, string>({{"route", "QFNS288"},
                             {"lot_number", "P23ASEA02"},
                             {"pin_package", "DFN-08YM2G"},
                             {"bd_id", "AAS008YM2024A"},
                             {"prod_id", "008YMAS034"},
                             {"urgent_code", "urgent"},
                             {"customer", ""},
                             {"wb_location", ""},
                             {"qty", "16000"},
                             {"oper", "2200"},
                             {"hold", "N"},
                             {"mvin", "Y"},
                             {"sub_lot", "9"},
                             {"queue_time", "240.345"},
                             {"fcst_time", "123.2"},
                             {"amount_of_tools", "10"},
                             {"amount_of_wires", "20"},
                             {"CAN_RUN_MODELS", "UTC1000S,UTC2000S,UTC3000S"},
                             {"PROCESS_TIME", "123.45,456.78"},
                             {"uphs", "23,45,67"},
                             {"part_id", "PART_ID"},
                             {"part_no", "PART_NO"}});

    test_lot_csv_data_3 =
        map<string, string>({{"route", "QFNS288"},
                             {"lot_number", "P23ASEA02"},
                             {"pin_package", "DFN-08YM2G"},
                             {"bd_id", "AAS008YM2024A"},
                             {"prod_id", "008YMAS034"},
                             {"urgent_code", "urgent"},
                             {"customer", ""},
                             {"wb_location", ""},
                             {"qty", "16000"},
                             {"oper", "2200"},
                             {"hold", "N"},
                             {"mvin", "Y"},
                             {"sub_lot", "9"},
                             {"queue_time", "240.345"},
                             {"fcst_time", "123.2"},
                             {"amount_of_tools", "10"},
                             {"amount_of_wires", "20"},
                             {"CAN_RUN_MODELS", "UTC1000S,UTC2000S,UTC3000S"},
                             {"PROCESS_TIME", "123.45,456.78,789.1"},
                             {"uphs", "23,45"},
                             {"part_id", "PART_ID"},
                             {"part_no", "PART_NO"}});
}

TEST_F(test_lot_t, test_lot_ctor_wip_data1)
{
    lot = new lot_t(test_wip_data_1);
    if (lot == nullptr)
        exit(EXIT_FAILURE);
    EXPECT_EQ(lot->_route.compare("QFNS288"), 0);
    EXPECT_EQ(lot->_lot_number.compare("P23ASEA02"), 0);
    EXPECT_EQ(lot->_pin_package.compare("DFN-08YM2G"), 0);
    EXPECT_EQ(lot->_recipe.compare("AAS008YM2024A"), 0);
    EXPECT_EQ(lot->_prod_id.compare("008YMAS034"), 0);
    EXPECT_EQ(lot->_urgent.compare("urgent"), 0);
    EXPECT_EQ(lot->_customer.compare("ICSI"), 0);
    EXPECT_EQ(lot->_wb_location.compare("BB211-1"), 0);
    EXPECT_EQ(lot->_qty, 16000);
    EXPECT_EQ(lot->_oper, 2200);
    EXPECT_EQ(lot->_hold, false);
    EXPECT_EQ(lot->_mvin, true);
    EXPECT_EQ(lot->_sub_lots, 9);
    EXPECT_EQ(lot->_is_sub_lot, true);

    EXPECT_EQ(lot->_prescheduled_order, 1);
    EXPECT_EQ(lot->_prescheduled_machine.compare("BB211"), 0);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_ctor_wip_data2)
{
    lot = new lot_t(test_wip_data_2);
    if (lot == nullptr)
        exit(EXIT_FAILURE);
    EXPECT_EQ(lot->_lot_number.compare("P23ASEA"), 0);
    EXPECT_EQ(lot->_is_sub_lot, false);
    EXPECT_EQ(lot->_qty, 0);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_ctor_wip_data3)
{
    lot = new lot_t(test_wip_data_3);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    EXPECT_EQ(lot->_wb_location.compare("BB211"), 0);
    EXPECT_EQ(lot->_prescheduled_machine.compare("BB211"), 0);
    EXPECT_EQ(lot->_prescheduled_order, 0)
        << "Prescheduled order :" << lot->_prescheduled_order << endl;

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_ctor_csv_data1)
{
    lot = new lot_t(test_lot_csv_data_1);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    double uphs[] = {23, 45, 67};
    double process_times[] = {123.45, 456.78, 789.1};

    EXPECT_EQ(lot->_can_run_models.size(), 3);
    EXPECT_EQ(lot->_can_run_models[0].compare("UTC1000S"), 0);
    EXPECT_EQ(lot->_can_run_models[1].compare("UTC2000S"), 0);
    EXPECT_EQ(lot->_can_run_models[2].compare("UTC3000S"), 0);

    for (unsigned int i = 0; i < lot->_can_run_models.size(); ++i) {
        EXPECT_NEAR(lot->_model_process_times[lot->_can_run_models[i]],
                    process_times[i], 0.0000001);
    }

    for (unsigned int i = 0; i < lot->_can_run_models.size(); ++i) {
        EXPECT_NEAR(lot->_uphs[lot->_can_run_models[i]], uphs[i], 0.0000001);
    }

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_ctor_csv_data2)
{
    EXPECT_THROW(new lot_t(test_lot_csv_data_2), invalid_argument);
}

TEST_F(test_lot_t, test_lot_ctor_csv_data3)
{
    EXPECT_THROW(new lot_t(test_lot_csv_data_3), invalid_argument);
}


TEST_F(test_lot_t, test_lot_isPrescheduled_true1)
{
    lot = new lot_t(test_wip_data_1);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    EXPECT_EQ(lot->isPrescheduled(), true);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_isPrescheduled_true2)
{
    lot = new lot_t(test_wip_data_3);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    EXPECT_EQ(lot->isPrescheduled(), true);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_isPrescheduled_false1)
{
    lot = new lot_t(test_wip_data_4);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    EXPECT_EQ(lot->isPrescheduled(), false);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_isPrescheduled_false2)
{
    lot = new lot_t(test_wip_data_5);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    EXPECT_EQ(lot->isPrescheduled(), false);

    delete lot;
    lot = nullptr;
}
