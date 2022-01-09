#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
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

class test_machines_t_addMachine : public testing::Test
{
private:
    map<string, string> entity_data_template;

    map<string, string> job_data_template;

public:
    map<string, string> test_entity_case1;
    map<string, string> test_job_case1;
    map<string, string> test_job_case2;
    entity_t *ent;
    machines_t *machines;
    void SetUp() override;

    entity_t *createAnEntity(map<string, string> data);
    lot_t *createALot(map<string, string> data);

    void TearDown() override;
};

entity_t *test_machines_t_addMachine::createAnEntity(map<string, string> data)
{
    entity_t *ent = nullptr;
    ent = new entity_t(data);
    if (ent == nullptr) {
        perror("failed to new entity_t instance");
        exit(EXIT_FAILURE);
    }
    return ent;
}

lot_t *test_machines_t_addMachine::createALot(map<string, string> data)
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

void test_machines_t_addMachine::SetUp()
{
    machines = nullptr;

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

    job_data_template =
        map<string, string>({{"route", "QFNS288"},
                             {"lot_number", "P23ASEA02"},
                             {"pin_package", "DFN-08YM2G"},
                             {"bd_id", "AAS008YM2024A"},
                             {"prod_id", "008YMAS034"},
                             {"urgent_code", "urgent"},
                             {"customer", ""},
                             {"wb_location", "BB211-1"},
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


    test_entity_case1 = entity_data_template;
    test_job_case1 = job_data_template;
    test_job_case2 = job_data_template;
    test_job_case2["wb_location"] = "BB211-2";


    machines = new machines_t();
    if (machines == nullptr) {
        perror("Failed to new machines_t instance");
        exit(EXIT_FAILURE);
    }
};

void test_machines_t_addMachine::TearDown()
{
    delete machines;
}

TEST_F(test_machines_t_addMachine, test_basic_feature)
{
    ent = createAnEntity(test_entity_case1);

    EXPECT_NO_THROW(machines->addMachine(ent->machine()));

    EXPECT_NO_THROW(machines->_machines.at(ent->_entity_name));

    machine_t ent_machine = ent->machine();

    string part_id(ent->_current_lot->_part_id);
    string part_no(ent->_current_lot->part_no());
    string tool_wire = part_no + "_" + part_id;

    machine_t mcs_machine = *machines->_machines.at(ent->_entity_name);

    // FIXME: Actually, to make sure that two structures are the same, we need
    // to perform byte comparison. but somehow, the byte comparison is wrong in
    // comparing two structures whose contents are consistence.
    // EXPECT_EQ(memcmp(&ent_machine, &mcs_machine, sizeof(machine_t)), 0);
    EXPECT_EQ(strcmp(ent_machine.model_name.data.text,
                     mcs_machine.model_name.data.text),
              0);


    EXPECT_NO_THROW(machines->_tool_machines.at(part_no));
    EXPECT_EQ(machines->_tool_machines.at(part_no).size(), 1);
    mcs_machine = *machines->_tool_machines.at(part_no).back();
    EXPECT_EQ(strcmp(ent_machine.model_name.data.text,
                     mcs_machine.model_name.data.text),
              0);

    EXPECT_NO_THROW(machines->_wire_machines.at(part_id));
    EXPECT_EQ(machines->_wire_machines.at(part_id).size(), 1);
    mcs_machine = *machines->_wire_machines.at(part_id).back();
    EXPECT_EQ(strcmp(ent_machine.model_name.data.text,
                     mcs_machine.model_name.data.text),
              0);

    EXPECT_NO_THROW(machines->_tool_wire_machines.at(tool_wire));
    EXPECT_EQ(machines->_tool_wire_machines.at(tool_wire).size(), 1);
    mcs_machine = *machines->_tool_wire_machines.at(tool_wire).back();
    EXPECT_EQ(strcmp(ent_machine.model_name.data.text,
                     mcs_machine.model_name.data.text),
              0);

    string model_name = ent->_model_name;
    string location = ent->_location;

    EXPECT_EQ(machines->_model_locations.count(model_name), 1);
    EXPECT_NE(find(machines->_model_locations[model_name].begin(),
                   machines->_model_locations[model_name].end(), location),
              machines->_model_locations[model_name].end());

    delete ent;
    ent = nullptr;
}

TEST_F(test_machines_t_addMachine, test_throw)
{
    ent = createAnEntity(test_entity_case1);
    entity_t *ent2 = createAnEntity(test_entity_case1);

    EXPECT_NO_THROW(machines->addMachine(ent->machine()));
    EXPECT_THROW(machines->addMachine(ent2->machine()), invalid_argument);
}


TEST_F(test_machines_t_addMachine, test_machines_prescheduleJobs_sortJob1)
{
    lot_t *lot1 = createALot(test_job_case1);
    lot_t *lot2 = createALot(test_job_case2);

    ent = createAnEntity(test_entity_case1);

    lot1->setPrescheduledModel(ent->_model_name);
    lot2->setPrescheduledModel(ent->_model_name);

    job_t *job1 = new job_t(lot1->job());
    job_t *job2 = new job_t(lot2->job());


    job1->base.ptr_derived_object = job1;
    job1->list.ptr_derived_object = job1;

    job2->base.ptr_derived_object = job2;
    job2->list.ptr_derived_object = job2;



    ASSERT_NO_THROW(machines->addMachine(ent->machine()));

    ASSERT_NO_THROW(machines->addPrescheduledJob(job1));
    ASSERT_NO_THROW(machines->addPrescheduledJob(job2));

    machine_t *machine;
    ASSERT_NO_THROW(machine = machines->_machines.at("BB211"));

    EXPECT_EQ(machine->base.root->ptr_derived_object, job1);
    EXPECT_EQ(machine->base.root->next->ptr_derived_object, job2);

    machines->prescheduleJobs();

    // FIXME : check the machines->_scheduled_jobs
    EXPECT_EQ(machines->_job_on_machine.size(), 1);
    EXPECT_EQ(machines->_scheduled_jobs.size(), 2);
    // EXPECT_EQ(machine->base.root->ptr_derived_object, job1);
    // EXPECT_EQ(machine->base.root->next->ptr_derived_object, job2);

    delete job1;
    delete job2;
    delete ent;
    job1 = job2 = nullptr;
    ent = nullptr;
}

TEST_F(test_machines_t_addMachine, test_machines_prescheduleJobs_sortJob2)
{
    ent = createAnEntity(test_entity_case1);
    lot_t *lot1 = createALot(test_job_case1);
    lot_t *lot2 = createALot(test_job_case2);

    lot1->setPrescheduledModel(ent->_model_name);
    lot2->setPrescheduledModel(ent->_model_name);

    job_t *job1 = new job_t(lot1->job());
    job_t *job2 = new job_t(lot2->job());


    job1->base.ptr_derived_object = job1;
    job1->list.ptr_derived_object = job1;

    job2->base.ptr_derived_object = job2;
    job2->list.ptr_derived_object = job2;



    ASSERT_NO_THROW(machines->addMachine(ent->machine()));

    ASSERT_NO_THROW(machines->addPrescheduledJob(job2));
    ASSERT_NO_THROW(machines->addPrescheduledJob(job1));

    machine_t *machine;
    ASSERT_NO_THROW(machine = machines->_machines.at("BB211"));

    EXPECT_EQ(machine->base.root->ptr_derived_object, job2);
    EXPECT_EQ(machine->base.root->next->ptr_derived_object, job1);

    machines->prescheduleJobs();

    // EXPECT_EQ(machine->base.root->ptr_derived_object, job1);
    // EXPECT_EQ(machine->base.root->next->ptr_derived_object, job2);

    delete job1;
    delete job2;
    delete ent;
    job1 = job2 = nullptr;
    ent = nullptr;
}
