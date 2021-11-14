#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include "include/info.h"
#include "include/infra.h"


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
    entity_data = map<string, string>{
        {      "entity",         "BB211"},
        {       "model",       "UTC3000"},
        {"recover_time", "21-6-19 15:24"},
        {     "in_time", "21-6-19 14:24"},
        {     "prod_id",    "048TPAW086"},
        { "pin_package",   "TSOP1-48/M2"},
        {  "lot_number",     "P23AWDV31"},
        {    "customer",          "MXIC"},
        {       "bd_id", "AAW048TP1041B"},
        {        "oper",          "2200"},
        {         "qty",          "1280"},
        {    "location",          "TA-A"},
        {     "part_no",       "PART_NO"},
        {     "part_id",       "PART_ID"}
    };

    ent = new entity_t(entity_data, timeConverter("21-6-19 14:34"));
    if (ent == nullptr) {
        perror("new entity_t instance");
        exit(EXIT_FAILURE);
    }


    lot_data_template = map<string, string>({
        {      "route",       "QFNS288"},
        { "lot_number",     "P23ASEA02"},
        {"pin_package",    "DFN-08YM2G"},
        {      "bd_id", "AAS008YM2024A"},
        {    "prod_id",    "008YMAS034"},
        {"urgent_code",        "urgent"},
        {   "customer",          "MXIC"},
        {"wb_location",       "BB211-1"},
        {        "qty",         "16000"},
        {       "oper",          "2200"},
        {       "hold",             "N"},
        {       "mvin",             "Y"},
        {    "sub_lot",             "9"},
        {    "part_id",       "PART_ID"},
        {    "part_no",       "PART_NO"}
    });

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
    EXPECT_EQ(ent->_current_lot->_lot_number.compare("P23AWDV31"), 0);
    EXPECT_EQ(ent->_current_lot->_pin_package.compare("TSOP1-48/M2"), 0);
    EXPECT_EQ(ent->_current_lot->_recipe.compare("AAW048TP1041B"), 0);
    EXPECT_EQ(ent->_current_lot->_customer.compare("MXIC"), 0);
    EXPECT_EQ(ent->_current_lot->_qty, 1280);
    EXPECT_EQ(ent->_current_lot->_oper, 2200);
    EXPECT_EQ(ent->_current_lot->_part_id.compare("PART_ID"), 0);
    EXPECT_EQ(ent->_current_lot->part_no().compare("PART_NO"), 0);
}


TEST_F(test_entity_t, test_entity_machine_basic_content)
{
    machine_t machine = ent->machine();

    EXPECT_EQ(
        strcmp(machine.base.machine_no.data.text, ent->_entity_name.c_str()),
        0);
    EXPECT_EQ(machine.base.size_of_jobs, 0);
    EXPECT_EQ(machine.base.available_time, ent->_recover_time);

    EXPECT_EQ(strcmp(machine.model_name.data.text, ent->_model_name.c_str()),
              0);
    EXPECT_EQ(strcmp(machine.location.data.text, ent->_location.c_str()), 0);
    // EXPECT_EQ(machine.tool, nullptr);
    // EXPECT_EQ(machine.wire, nullptr);
    EXPECT_EQ(machine.makespan, 0);
    EXPECT_EQ(machine.total_completion_time, 0);
    EXPECT_EQ(machine.quality, 0);
    EXPECT_EQ(machine.setup_times, 0);
    EXPECT_EQ(machine.ptr_derived_object, nullptr);
}

TEST_F(test_entity_t, test_entity_machine_current_job)
{
    machine_t machine = ent->machine();

    job_t job = machine.current_job;

    EXPECT_EQ(job.oper, 2200);
    EXPECT_EQ(strcmp(job.bdid.data.text, "AAW048TP1041B"), 0);
    EXPECT_EQ(strcmp(job.customer.data.text, "MXIC"), 0);
    EXPECT_EQ(strcmp(job.part_id.data.text, "PART_ID"), 0);
    EXPECT_EQ(strcmp(job.part_no.data.text, "PART_NO"), 0);
    EXPECT_EQ(strcmp(job.pin_package.data.text, "TSOP1-48/M2"), 0);
    EXPECT_EQ(strcmp(job.prod_id.data.text, "048TPAW086"), 0);

    EXPECT_EQ(job.base.start_time, -10);
    EXPECT_EQ(job.base.end_time, ent->_recover_time);
    EXPECT_EQ(job.base.qty, 1280);
    EXPECT_EQ(strcmp(job.base.job_info.data.text, "P23AWDV31"), 0);
}
