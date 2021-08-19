#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>

#include "include/entities.h"
#include "include/entity.h"
#include "include/infra.h"
#include "include/lot.h"
#include "include/machine.h"


using namespace std;

entity_t::entity_t(map<string, string> elements)
{
    _current_lot = nullptr;
    _outplan_time = _recover_time = timeConverter(elements["recover_time"]);
    _entity_name = elements["entity"];
    _model_name = elements["model"];
    _location = elements["location"];
    _current_lot = new lot_t(elements);

    if (_current_lot == nullptr) {
        perror("new current_lot error");
        exit(EXIT_FAILURE);
    }
}

machine_t entity_t::machine()
{
    machine_t machine =
        machine_t{.base = {.machine_no = stringToInfo(_entity_name),
                           .size_of_jobs = 0,
                           .available_time = _recover_time},
                  .model_name = stringToInfo(_model_name),
                  .location = stringToInfo(_location),
                  .tool = nullptr,
                  .wire = nullptr,
                  .current_job = _current_lot->job(),
                  .makespan = 0,
                  .total_completion_time = 0,
                  .quality = 0,
                  .setup_times = 0,
                  .ptr_derived_object = nullptr};

    machine.current_job.base.end_time = _recover_time;

    return machine;
}


bool entityComparisonByTime(entity_t *ent1, entity_t *ent2)
{
    return ent1->getRecoverTime() < ent2->getRecoverTime();
}


unsigned int convertEntityNameToUInt(string name)
{
    union {
        char text[4];
        unsigned int number;
    } data;
    string substr = name.substr(name.length() - 4);
    strncpy(data.text, substr.c_str(), 4);
    return data.number;
}



ancillary_resources_t::ancillary_resources_t(std::map<std::string, int> data)
{
    tool_t *t;
    for (std::map<std::string, int>::iterator it = data.begin();
         it != data.end(); it++) {
        for (int i = 0; i < it->second; ++i) {
            t = new tool_t;
            t->time = 0;
            t->machine_no = 0;
            _tools[it->first].push_back(t);
        }
    }
}

std::vector<tool_t *> ancillary_resources_t::aRound(
    std::map<std::string, int> amounts)
{
    std::vector<tool_t *> ts;
    for (std::map<std::string, int>::iterator it = amounts.begin();
         it != amounts.end(); it++) {
        std::vector<tool_t *> tmp = aRound(it->first, it->second);
        ts += tmp;
    }

    return ts;
}

std::vector<tool_t *> ancillary_resources_t::aRound(std::string name,
                                                    int amount)
{
    sort(_tools[name].begin(), _tools[name].end(), aresPtrComp);
    std::vector<tool_t *> ts(_tools[name].begin(),
                             _tools[name].begin() + amount);
    return ts;
}



std::string convertUIntToEntityName(unsigned int mno)
{
    std::string text = "B";
    union {
        char text[5];
        unsigned int number;
    } data;

    data.number = mno;
    data.text[4] = '\0';
    text += data.text;
    return text;
}
