#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include "include/info.h"
#include "include/infra.h"
#include "include/job.h"
#include "include/job_base.h"

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
    map<string, string> test_lot_csv_data_4;


    lot_t *lot;

    lot_t *createALot(map<string, string> elements);

    void SetUp() override;

    void TearDown() override;
};

lot_t *test_lot_t::createALot(map<string, string> elements)
{
    lot_t *lot = nullptr;
    lot = new lot_t(elements);
    if (lot == nullptr)
        exit(EXIT_FAILURE);
    return lot;
}

void test_lot_t::TearDown()
{
    if (lot != nullptr)
        delete lot;
    lot = nullptr;
}

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

    test_wip_data_5 = map<string, string>({
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
                             {"recipe", "AAS008YM2024A"},
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
                             {"recipe", "AAS008YM2024A"},
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
                             {"recipe", "AAS008YM2024A"},
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

    test_lot_csv_data_4 = test_lot_csv_data_1;
    test_lot_csv_data_4["wb_location"] = "BB211-1";
}

TEST_F(test_lot_t, test_lot_default_ctor)
{
    lot = new lot_t();
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    EXPECT_EQ(lot->_status, SUCCESS);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_ctor_wip_data1)
{
    lot = createALot(test_wip_data_1);

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

    EXPECT_EQ(lot->_status, SUCCESS);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_ctor_wip_data2)
{
    lot = createALot(test_wip_data_2);
    EXPECT_EQ(lot->_lot_number.compare("P23ASEA"), 0);
    EXPECT_EQ(lot->_is_sub_lot, false);
    EXPECT_EQ(lot->_qty, 0);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_ctor_wip_data3)
{
    lot = createALot(test_wip_data_3);

    EXPECT_EQ(lot->_wb_location.compare("BB211"), 0);
    EXPECT_EQ(lot->_prescheduled_machine.compare("BB211"), 0);
    EXPECT_EQ(lot->_prescheduled_order, 0)
        << "Prescheduled order :" << lot->_prescheduled_order << endl;

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_ctor_csv_data1)
{
    lot = createALot(test_lot_csv_data_1);

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
    lot = createALot(test_wip_data_1);

    EXPECT_EQ(lot->isPrescheduled(), true);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_isPrescheduled_true2)
{
    lot = createALot(test_wip_data_3);

    EXPECT_EQ(lot->isPrescheduled(), true);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_isPrescheduled_false1)
{
    lot = createALot(test_wip_data_4);

    EXPECT_EQ(lot->isPrescheduled(), false);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_isPrescheduled_false2)
{
    lot = createALot(test_wip_data_5);
    if (lot == nullptr)
        exit(EXIT_FAILURE);

    EXPECT_EQ(lot->isPrescheduled(), false);

    delete lot;
    lot = nullptr;
}

TEST_F(test_lot_t, test_lot_job1)
{
    lot = createALot(test_lot_csv_data_1);
    job_t job = lot->job();

    EXPECT_EQ(strcmp(job.base.job_info.data.text, lot->_lot_number.c_str()), 0);

    EXPECT_EQ(strcmp(job.part_no.data.text, lot->part_no().c_str()), 0);
    EXPECT_EQ(strcmp(job.part_id.data.text, lot->_part_id.c_str()), 0);
    EXPECT_EQ(strcmp(job.prod_id.data.text, lot->_prod_id.c_str()), 0);

    EXPECT_EQ(strcmp(job.pin_package.data.text, lot->_pin_package.c_str()), 0);
    EXPECT_EQ(strcmp(job.customer.data.text, lot->_customer.c_str()), 0);
    EXPECT_EQ(strcmp(job.bdid.data.text, lot->_recipe.c_str()), 0);

    EXPECT_EQ(job.oper, lot->tmp_oper);

    EXPECT_EQ(job.base.qty, lot->_qty);
    EXPECT_EQ(job.base.start_time, 0);
    EXPECT_EQ(job.base.end_time, 0);
    EXPECT_EQ(job.base.arriv_t, lot->_queue_time);
    EXPECT_EQ(job.is_scheduled, false);

    info_t machine_no = job.base.machine_no;
    info_t empty_info = emptyInfo();
    EXPECT_EQ(memcmp(&machine_no, &empty_info, sizeof(info_t)), 0);
}

TEST_F(test_lot_t, test_lot_job2)
{
    lot = createALot(test_lot_csv_data_4);  // which has wb_location field
    string machine_no("BB211");
    info_t mc_info = stringToInfo(machine_no);

    job_t job = lot->job();
    EXPECT_EQ(isSameInfo(job.base.machine_no, mc_info), true);
    EXPECT_EQ(job.weight, 1);
    EXPECT_EQ(job.list.get_value, prescheduledJobGetValue);
    EXPECT_NEAR(job.base.ptime, 123.45, 0.00001);
}

TEST_F(test_lot_t, test_lot_job3)
{
    lot = createALot(test_lot_csv_data_4);
    lot->setPrescheduledModel("UTC1000S");
    job_t job = lot->job();
    info_t empty_info = emptyInfo();
    EXPECT_NE(strcmp(job.bdid.data.text, empty_info.data.text), 0);
    EXPECT_NEAR(job.base.ptime, 123.45, 0.000001);
}

TEST_F(test_lot_t, test_lot_isInSchedulingPlan)
{
    test_lot_csv_data_4["lot_number"] = "PXXABK07";
    lot = createALot(test_lot_csv_data_4);
    EXPECT_FALSE(lot->isInSchedulingPlan());

    lot = createALot(test_lot_csv_data_1);
    EXPECT_TRUE(lot->isInSchedulingPlan());
}
