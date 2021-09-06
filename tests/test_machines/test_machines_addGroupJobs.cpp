#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include "include/info.h"
#include "include/job.h"

#define private public
#define protected public

#include "include/entity.h"
#include "include/lot.h"
#include "include/machine.h"
#include "include/machines.h"

#undef private
#undef protected

using namespace std;

class test_machines_t_addGroupsJobs : public testing::Test
{
private:
    map<string, string> entity_data_template;

    map<string, string> lot_data_template;

public:
    map<string, string> test_entity_case1;
    map<string, string> test_entity_case2;
    map<string, string> test_entity_case3;
    map<string, string> test_entity_case4;

    map<string, string> test_lot_case1;
    map<string, string> test_lot_case2;
    map<string, string> test_lot_case3;
    map<string, string> test_lot_case4;

    entity_t *createAnEntity(map<string, string> data);
    lot_t *createALot(map<string, string> data);

    machines_t *machines;

    entity_t *ent1, *ent2, *ent3, *ent4;
    lot_t *lot1, *lot2, *lot3, *lot4;

    void SetUp() override;
    void TearDown() override;
};

entity_t *test_machines_t_addGroupsJobs::createAnEntity(
    map<string, string> data)
{
    entity_t *ent = nullptr;
    ent = new entity_t(data);
    if (ent == nullptr) {
        perror("failed to new entity_t instance");
        exit(EXIT_FAILURE);
    }
    return ent;
}

lot_t *test_machines_t_addGroupsJobs::createALot(map<string, string> data)
{
    lot_t *lot = nullptr;
    lot = new lot_t(data);
    if (lot == nullptr) {
        perror(
            "Failed to new lot_t instalce in "
            "test_machines_t_addMachine::createALot");
        exit(EXIT_FAILURE);
    }

    return lot;
}

void test_machines_t_addGroupsJobs::SetUp()
{
    machines = nullptr;
    machines = new machines_t();
    if (machines == nullptr) {
        perror(
            "Failed to allocate memory in "
            "test_machines_t_addGroupsJobs::SetUp");
        exit(EXIT_FAILURE);
    }


    entity_data_template =
        map<string, string>{{"entity", "BB211"},
                            {"model", "UTC3000"},
                            {"recover_time", "2021/6/19 15:24"},
                            {"prod_id", "048TPAW086"},
                            {"pin_package", "TSOP1-48/M2"},
                            {"lot_number", "P23AWDV31"},
                            {"customer", "MXIC"},
                            {"bd_id", "AAS008YM2024A"},
                            {"oper", "2200"},
                            {"qty", "1280"},
                            {"location", "TA-A"},
                            {"part_no", "PART_NO"},
                            {"part_id", "PART_ID"}};

    test_entity_case1 = entity_data_template;
    test_entity_case2 = entity_data_template;
    test_entity_case2["entity"] = "BB212";

    test_entity_case3 = entity_data_template;
    test_entity_case3["entity"] = "BB213";
    test_entity_case3["bd_id"] = "BDID";

    test_entity_case4 = test_entity_case3;
    test_entity_case4["entity"] = "BB214";


    ent1 = createAnEntity(test_entity_case1);
    ent2 = createAnEntity(test_entity_case2);
    ent3 = createAnEntity(test_entity_case3);
    ent4 = createAnEntity(test_entity_case4);


    lot_data_template =
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
                             {"CAN_RUN_MODELS", "UTC1000S,UTC2000S,UTC3000"},
                             {"PROCESS_TIME", "123.45,456.78,789.1"},
                             {"uphs", "23,45,67"},
                             {"part_id", "PART_ID"},
                             {"part_no", "PART_NO"}});


    test_lot_case1 = lot_data_template;
    test_lot_case2 = lot_data_template;
    test_lot_case2["lot_number"] = "lot2";

    test_lot_case3 = lot_data_template;
    test_lot_case3["bd_id"] = "BDID";
    test_lot_case3["lot_number"] = "lot3";

    test_lot_case4 = test_lot_case3;
    test_lot_case4["lot_number"] = "lot4";

    lot1 = createALot(test_entity_case1);
    lot2 = createALot(test_entity_case2);
    lot3 = createALot(test_entity_case3);
    lot4 = createALot(test_entity_case4);
}

void test_machines_t_addGroupsJobs::TearDown()
{
    delete ent1;
    delete ent2;
    delete ent3;
    delete ent4;
    delete lot1;
    delete lot2;
    delete lot3;
    delete lot4;
}

TEST_F(test_machines_t_addGroupsJobs, machines_collection)
{
    machines->addMachine(ent1->machine());
    machines->addMachine(ent2->machine());
    machines->addMachine(ent3->machine());
    machines->addMachine(ent4->machine());

    job_t *j1 = new job_t();
    *j1 = lot1->job();

    job_t *j2 = new job_t();
    *j2 = lot2->job();

    vector<job_t *> g1 = {j1, j2};

    job_t *j3 = new job_t();
    *j3 = lot3->job();

    job_t *j4 = new job_t();
    *j4 = lot4->job();

    vector<job_t *> g2 = {j3, j4};

    machines->addGroupJobs(lot1->_recipe, g1);
    machines->addGroupJobs(lot3->_recipe, g2);

    EXPECT_EQ(machines->_dispatch_groups.count("AAS008YM2024A"), 1);
    EXPECT_EQ(machines->_dispatch_groups.count("BDID"), 1);

    EXPECT_EQ(machines->_dispatch_groups["AAS008YM2024A"]->unscheduled_jobs[0],
              j1);
    EXPECT_EQ(machines->_dispatch_groups["AAS008YM2024A"]->unscheduled_jobs[1],
              j2);
    EXPECT_EQ(strcmp(machines->_dispatch_groups["AAS008YM2024A"]
                         ->machines[0]
                         ->base.machine_no.data.text,
                     ent1->_entity_name.c_str()),
              0);
    EXPECT_EQ(strcmp(machines->_dispatch_groups["AAS008YM2024A"]
                         ->machines[1]
                         ->base.machine_no.data.text,
                     ent2->_entity_name.c_str()),
              0);


    EXPECT_EQ(machines->_dispatch_groups["BDID"]->unscheduled_jobs[0], j3);
    EXPECT_EQ(machines->_dispatch_groups["BDID"]->unscheduled_jobs[1], j4);
    EXPECT_EQ(strcmp(machines->_dispatch_groups["BDID"]
                         ->machines[0]
                         ->base.machine_no.data.text,
                     ent3->_entity_name.c_str()),
              0);
    EXPECT_EQ(strcmp(machines->_dispatch_groups["BDID"]
                         ->machines[1]
                         ->base.machine_no.data.text,
                     ent4->_entity_name.c_str()),
              0);
}
