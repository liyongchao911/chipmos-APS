//
// Created by YuChunLin on 2021/8/21.
//

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

class test_machines_t_prescheduleJobs : public testing::Test
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
    lot_t *lot1, *lot2;
    job_t *job1, *job2;

    void SetUp() override;

    entity_t *createAnEntity(map<string, string> data);
    lot_t *createALot(map<string, string> data);

    void TearDown() override;
};

entity_t *test_machines_t_prescheduleJobs::createAnEntity(
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

lot_t *test_machines_t_prescheduleJobs::createALot(map<string, string> data)
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


void test_machines_t_prescheduleJobs::SetUp()
{
    machines = nullptr;

    entity_data_template =
        map<string, string>{{"entity", "BB211"},
                            {"model", "UTC3000"},
                            {"recover_time", "21-6-19 15:24"},
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
    test_job_case2["lot_number"] = "LOTNUMBER";


    machines = new machines_t();
    if (machines == nullptr) {
        perror("Failed to new machines_t instance");
        exit(EXIT_FAILURE);
    }
};

void test_machines_t_prescheduleJobs::TearDown()
{
    delete machines;
}

TEST_F(test_machines_t_prescheduleJobs, sorting_correctness1)
{
    ent = createAnEntity(test_entity_case1);
    lot1 = createALot(test_job_case1);
    lot2 = createALot(test_job_case2);

    lot1->setPrescheduledModel(ent->_model_name);
    lot2->setPrescheduledModel(ent->_model_name);

    job1 = new job_t(lot1->job());
    job2 = new job_t(lot2->job());


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

    // FIXME : check the machines->_scheduled_jobs
    // EXPECT_EQ(machine->base.root->ptr_derived_object, job1);
    // EXPECT_EQ(machine->base.root->next->ptr_derived_object, job2);


    EXPECT_NEAR(job1->base.end_time - job1->base.start_time, job1->base.ptime,
                0.001);
    EXPECT_NEAR(job2->base.end_time - job2->base.start_time, job2->base.ptime,
                0.001);
    EXPECT_NEAR(job2->base.start_time, job1->base.end_time, 0.000001);

    delete job1;
    delete job2;
    delete ent;
    job1 = job2 = nullptr;
    ent = nullptr;
}

TEST_F(test_machines_t_prescheduleJobs, sorting_correctness2)
{
    ent = createAnEntity(test_entity_case1);
    lot_t *lot1 = createALot(test_job_case2);
    lot_t *lot2 = createALot(test_job_case1);

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

    // FIXME : check the scheduled job
    // EXPECT_EQ(machine->base.root->ptr_derived_object, job2);
    // EXPECT_EQ(machine->base.root->next->ptr_derived_object, job1);


    EXPECT_NEAR(job1->base.end_time - job1->base.start_time, job1->base.ptime,
                0.001);
    EXPECT_NEAR(job2->base.end_time - job2->base.start_time, job2->base.ptime,
                0.001);
    EXPECT_NEAR(job1->base.start_time, job2->base.end_time, 0.000001);

    delete job1;
    delete job2;
    delete ent;
    job1 = job2 = nullptr;
    ent = nullptr;
}

TEST_F(test_machines_t_prescheduleJobs,
       test_last_job_of_machine_and_scheduled_feature)
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

    EXPECT_EQ(job1->is_scheduled, false);
    EXPECT_EQ(job2->is_scheduled, false);

    machines->prescheduleJobs();

    // FIXME : check the scheduled job
    // EXPECT_EQ(machine->base.root->ptr_derived_object, job1);
    // EXPECT_EQ(machine->base.root->next->ptr_derived_object, job2);


    EXPECT_NEAR(job1->base.end_time - job1->base.start_time, job1->base.ptime,
                0.001);
    EXPECT_NEAR(job2->base.end_time - job2->base.start_time, job2->base.ptime,
                0.001);
    EXPECT_NEAR(job2->base.start_time, job1->base.end_time, 0.000001);

    EXPECT_EQ(job1->is_scheduled, true);
    EXPECT_EQ(job2->is_scheduled, true);



    // testing for the last job of machine
    EXPECT_EQ(strcmp(machine->current_job.base.job_info.data.text,
                     job2->base.job_info.data.text),
              0);
    EXPECT_NEAR(machine->current_job.base.end_time, job2->base.end_time,
                0.0001);
    EXPECT_NEAR(machine->base.available_time,
                machine->current_job.base.end_time, 0.0001);

    delete job1;
    delete job2;
    delete ent;
    job1 = job2 = nullptr;
    ent = nullptr;
}
