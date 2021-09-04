#include <cstdio>
#include <iterator>
#include <stdexcept>

#include "include/entity.h"
#include "include/infra.h"
#include "include/lot.h"
#include "include/machine.h"


using namespace std;

entity_t::entity_t()
{
    _outplan_time = _recover_time = 0;
}

entity_t::entity_t(map<string, string> elements, time_t base_time)
{
    _current_lot = nullptr;
    _outplan_time = _recover_time = timeConverter(elements["recover_time"]);
    _entity_name = elements["entity"];
    _model_name = elements["model"];
    _location = elements["location"];

    setBaseTime(base_time);

    string lot_number = elements["lot_number"];

    if (elements["qty"].length()) {
        time_t intime = timeConverter(elements["in_time"]);
        time_t outtime = timeConverter(elements["recover_time"]);
        double interval = (outtime - intime) / 60.0;
        int qty = stoi(elements["qty"]);
        if (_outplan_time <= 0) {
            elements["qty"] = to_string(0);
        }
        // elements["qty"] = to_string((_recover_time / interval) * qty);
    }
    _current_lot = new lot_t(elements);

    if (_current_lot == nullptr) {
        perror("new current_lot error");
        exit(EXIT_FAILURE);
    }
}

void entity_t::setBaseTime(time_t base_time)
{
    double tmp_time = _recover_time - base_time;
    _recover_time = ((tmp_time > 0) ? tmp_time : 0) / 60;

    _outplan_time = _recover_time;
}

machine_t entity_t::machine()
{
    machine_t machine =
        machine_t{.base = {.machine_no = stringToInfo(_entity_name),
                           .size_of_jobs = 0,
                           .available_time = _recover_time},
                  .model_name = stringToInfo(_model_name),
                  .location = stringToInfo(_location),
                  .current_job = _current_lot->job(),
                  .makespan = 0,
                  .total_completion_time = 0,
                  .quality = 0,
                  .setup_times = 0,
                  .ptr_derived_object = nullptr};

    machine.current_job.base.end_time = _recover_time;
    machine.current_job.base.machine_no = stringToInfo(_entity_name);

    return machine;
}
