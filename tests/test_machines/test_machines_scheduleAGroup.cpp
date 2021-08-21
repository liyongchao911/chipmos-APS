#include <gtest/gtest.h>
#include <ctime>
#include <string>

#define private public
#define protected public

#include "include/entities.h"
#include "include/entity.h"
#include "include/lot.h"
#include "include/lots.h"
#include "include/machine.h"
#include "include/machines.h"

#undef private
#undef protected

using namespace std;

class test_machines_t_scheduleAGroup : public testing::Test
{
private:
    map<string, string> entity_data_template;
    map<string, string> entity_data_bd1;
    map<string, string> entity_data_bd2;

    map<string, string> lot_data_template;

    map<string, string> lot_data_bd1;
    map<string, string> lot_data_bd2;

    vector<entity_t *> entities;
    map<string, vector<lot_t *> > recipe_lots;
    time_t base_time;

public:
    machines_t *machines;
    void SetUp() override;
    void TearDown() override;
};

void test_machines_t_scheduleAGroup::SetUp()
{
    base_time = timeConverter("2021/6/19 07:30");
    entity_data_template =
        map<string, string>{{"entity", "BB211"},
                            {"model", "UTC3000"},
                            {"recover_time", "2021/6/19 15:24"},
                            {"prod_id", "048TPAW086"},
                            {"pin_package", "TSOP1-48/M2"},
                            {"lot_number", "P23AWDV31"},
                            {"customer", "MXIC"},
                            {"bd_id", "BD1"},
                            {"oper", "2200"},
                            {"qty", "1280"},
                            {"location", "TA-3"},
                            {"part_no", "PART_NO"},
                            {"part_id", "PART_ID"}};

    entity_data_bd1 = entity_data_template;
    entity_data_bd2 = entity_data_template;
    entity_data_bd2["bd_id"] = "BD2";

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
                             {"CAN_RUN_MODELS", "UTC1000,UTC2000,UTC3000"},
                             {"PROCESS_TIME", "123.45,456.78,789.1"},
                             {"uphs", "23,45,67"},
                             {"part_id", "PART_ID"},
                             {"part_no", "PART_NO"}});

    lot_data_bd1 = lot_data_template;
    lot_data_bd1["bd_id"] = "BD1";

    lot_data_bd2 = lot_data_template;
    lot_data_bd2["bd_id"] = "BD2";


    machines = nullptr;
    machines = new machines_t();
    if (machines == nullptr) {
        perror(
            "Failed to new a machines_t instance in "
            "test_machines_t_scheduleAGroup");
        exit(EXIT_FAILURE);
    }


    // create bunch of entities
    for (unsigned int i = 0; i < 4; ++i) {
        entity_data_bd1["entity"] = "BB21" + to_string(i);
        entity_data_bd1["model"] = "UTC" + to_string(i % 3 + 1) + "000";
        entities.push_back(new entity_t(entity_data_bd1, base_time - i * 10));
    }

    for (unsigned int i = 0; i < 4; ++i) {
        entity_data_bd2["entity"] = "BB22" + to_string(i);
        entity_data_bd2["model"] = "UTC" + to_string(i % 3 + 1) + "000";
        entities.push_back(new entity_t(entity_data_bd2, base_time + i * 10));
    }

    for (unsigned int i = 0; i < entities.size(); ++i) {
        machines->addMachine(entities[i]->machine());
    }


    // create bunch of lots
    for (unsigned int i = 0; i < 8; ++i) {
        lot_data_bd1["lot_number"] = "P0" + to_string(i);
        lot_data_bd1["queue_time"] = to_string(0);
        recipe_lots["BD1"].push_back(new lot_t(lot_data_bd1));
    }

    for (unsigned int i = 0; i < 8; ++i) {
        lot_data_bd1["lot_number"] = "P1" + to_string(i);
        lot_data_bd2["queue_time"] = to_string(0);
        recipe_lots["BD2"].push_back(new lot_t(lot_data_bd1));
    }

    for (map<string, vector<lot_t *> >::iterator it = recipe_lots.begin();
         it != recipe_lots.end(); ++it) {
        vector<job_t *> jobs;
        for (unsigned int i = 0; i < it->second.size(); i++) {
            it->second[i]->setCanRunLocation(machines->getModelLocations());
            machines->addJobLocation(it->second[i]->lotNumber(),
                                     it->second[i]->getCanRunLocations());
            machines->addJobProcessTimes(it->second[i]->lotNumber(),
                                         it->second[i]->getModelProcessTimes());
            job_t *job = new job_t();
            *job = it->second[i]->job();
            job->base.ptr_derived_object = job;
            job->list.ptr_derived_object = job;
            jobs.push_back(job);
        }
        machines->addGroupJobs(it->first, jobs);
    }
}

void test_machines_t_scheduleAGroup::TearDown()
{
    for (unsigned int i = 0; i < entities.size(); ++i) {
        delete entities[i];
    }

    for (map<string, vector<lot_t *> >::iterator it = recipe_lots.begin();
         it != recipe_lots.end(); it++) {
        for (unsigned int i = 0; i < it->second.size(); ++i) {
            delete it->second[i];
        }
    }

    delete machines;
}

TEST_F(test_machines_t_scheduleAGroup, test_group_setting)
{
    EXPECT_EQ(machines->_groups.count("BD1"), 1);
    EXPECT_EQ(machines->_groups.count("BD2"), 1);

    EXPECT_EQ(machines->_groups.at("BD1").unscheduled_jobs.size(), 8);
    EXPECT_EQ(machines->_groups.at("BD2").unscheduled_jobs.size(), 8);

    EXPECT_EQ(machines->_groups.at("BD1").machines.size(), 4);
    EXPECT_EQ(machines->_groups.at("BD2").machines.size(), 4);
}


TEST_F(test_machines_t_scheduleAGroup, test_sorting)
{
    for (map<string, struct __machine_group_t>::iterator it =
             machines->_groups.begin();
         it != machines->_groups.end(); ++it) {
        vector<machine_t *> ms = machines->_sortedMachines(it->second.machines);
        vector<job_t *> js = machines->_sortedJobs(it->second.unscheduled_jobs);
        for (unsigned int i = 1; i < ms.size(); ++i) {
            EXPECT_LE(ms[i - 1]->base.available_time,
                      ms[i]->base.available_time);
        }

        for (unsigned int i = 1; i < js.size(); ++i) {
            EXPECT_LE(js[i - 1]->base.arriv_t, js[i]->base.arriv_t);
        }
    }
}

TEST_F(test_machines_t_scheduleAGroup, test_correctness1)
{
    // edit the time make all of jobs be able to be scheduled
    for (map<string, machine_t *>::iterator it = machines->_machines.begin();
         it != machines->_machines.end(); it++) {
        it->second->base.available_time = 0;
    }
    machines->_scheduleAGroup(&machines->_groups["BD1"]);
    machines->_scheduleAGroup(&machines->_groups["BD2"]);

    // check
    EXPECT_EQ(machines->_groups["BD1"].unscheduled_jobs.size(), 0);
    EXPECT_EQ(machines->_groups["BD2"].unscheduled_jobs.size(), 0);
    EXPECT_EQ(machines->_groups["BD1"].scheduled_jobs.size(), 8);
    EXPECT_EQ(machines->_groups["BD2"].scheduled_jobs.size(), 8);


    for (map<string, machine_t *>::iterator it = machines->_machines.begin();
         it != machines->_machines.end(); it++) {
        EXPECT_EQ(it->second->base.size_of_jobs, 2);
    }
}