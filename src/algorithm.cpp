//
// Created by YuChunLin on 2021/8/21.
//

#include "include/algorithm.h"

#include <vector>

using namespace std;


void prescheduling(machines_t *machines, lots_t *lots)
{
    vector<lot_t *> prescheduled_lots = lots->prescheduledLots();
    vector<job_t *> prescheduled_jobs;
    iter(prescheduled_lots, i)
    {
        job_t *job = new job_t();
        try {
            string prescheduled_model = machines->getModelByEntityName(
                prescheduled_lots[i]->preScheduledEntity());
            prescheduled_lots[i]->setPrescheduledModel(prescheduled_model);
            *job = prescheduled_lots[i]->job();
            machines->addPrescheduledJob(job);
            prescheduled_jobs.push_back(job);
        } catch (out_of_range &e) {
            delete job;
            lots->pushBackNotPrescheduledLot(prescheduled_lots[i]);
        }
    }

    machines->prescheduleJobs();
}

void stage2Scheduling(machines_t *machines, lots_t *lots)
{
    map<string, vector<lot_t *> > groups;
    groups = lots->getLotsRecipeGroups();

    for (auto it = groups.begin(); it != groups.end(); it++) {
        vector<job_t *> jobs;
        iter(it->second, i)
        {
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

    machines->scheduleGroups();
}


void stage3Scheduling(machines_t *machines, lots_t *lots)
{
    machines->setNumberOfTools(lots->amountOfTools());
    machines->setNumberOfWires(lots->amountOfWires());

    machines->groupJobsByToolAndWire();
    machines->distributeTools();
    machines->distributeWires();
    machines->chooseMachinesForGroups();
}