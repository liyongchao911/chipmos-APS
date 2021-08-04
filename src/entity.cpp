#include "include/entity.h"
#include <limits.h>
#include <sched.h>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include "include/infra.h"
#include "include/machine.h"


using namespace std;

machine_t entityToMachine(entity_t ent)
{
    return machine_t{
        .base = {.machine_no = convertEntityNameToUInt(ent.entity_name),
                 .size_of_jobs = 0,
                 .available_time = ent.recover_time},
        .tool = ent.tool,
        .wire = ent.wire,
        .current_job = ent.job,
        .quality = 0};
}

void entities_t::_readProcessIdFile(std::string filename)
{
    csv_t csv(filename, "r", true, true);
    csv.trim(" ");
    csv.setHeaders(map<string, string>({{"prod_id", "product"},
                                        {"process_id", "process_id"},
                                        {"bom_id", "bom_id"}}));

    map<string, string> row;
    for (int i = 0, nrows = csv.nrows(); i < nrows; ++i) {
        row = csv.getElements(i);
        prod_map_to_pid[row["prod_id"]] = row["process_id"];
        prod_map_to_bom_id[row["prod_id"]] = row["bom_id"];
    }
}

void entities_t::_readPartNoFile(std::string filename)
{
    csv_t csv(filename, "r", true, true);
    csv.trim(" ");
    csv.setHeaders(map<string, string>(
        {{"process_id", "process_id"}, {"remark", "remark"}}));
    map<string, string> pid_remark;
    map<string, string> row;
    string remark;
    for (int i = 0, size = csv.nrows(); i < size; ++i) {
        row = csv.getElements(i);
        remark = row["remark"];
        if (remark[0] == 'A') {
            remark = remark.substr(0, remark.find("("));
            if (remark.find("(") != std::string::npos) {
                remark = remark.substr(0, remark.find("("));
            }
            pid_remark[row["process_id"]] = remark;
        }
    }
    pid_map_to_part_no = pid_remark;
}

void entities_t::_readPartIdFile(std::string filename)
{
    csv_t csv(filename, "r", true, true);
    csv.trim(" ");
    csv.setHeaders(map<string, string>(
        {{"bom_id", "bom_id"}, {"oper", "oper"}, {"part_id", "part_id"}}));
    map<string, string> row;
    for (int i = 0, size = csv.nrows(); i < size; ++i) {
        row = csv.getElements(i);
        bom_id_map_to_part_id[row["bom_id"] + "_" + row["oper"]] =
            row["part_id"];
    }
}


entities_t::entities_t(std::map<std::string, std::string> arguments)
{
    _time = 0;
    setTime(arguments["std_time"]);
    _readProcessIdFile(arguments["pid_bomid"]);
    _readPartNoFile(arguments["pid_heatblock"]);
    _readPartIdFile(arguments["bom_list"]);
}

void entities_t::setTime(string time)
{
    if (time.length()) {
        _time = timeConverter(time);
    }
}


void entities_t::addMachine(map<string, string> elements)
{
    if (elements["recover_time"].length() == 0) {
        throw std::invalid_argument("recover time is empty");
    }

    double recover_time = timeConverter(elements["recover_time"]);


    entity_t *ent = new entity_t();
    string model, location;
    model = elements["model"];
    location = elements["location"];
    string prod_id;
    if (ent) {
        unsigned int no = convertEntityNameToUInt(elements["entity"]);
        prod_id = elements["prod_id"];
        string part_no = pid_map_to_part_no[prod_map_to_pid[prod_id]];
        string part_id = bom_id_map_to_part_id[prod_map_to_bom_id[prod_id] +
                                               "_" + elements["oper"]];
        job_t job =
            job_t{.part_no = stringToInfo(part_no),
                  .pin_package = stringToInfo(elements["pin_pkg"]),
                  .customer = stringToInfo(elements["customer"]),
                  .base = {.job_info = stringToInfo(elements["lot_number"]),
                           .qty = 0,
                           .machine_no = no,
                           .start_time = 0,
                           .end_time = recover_time},
                  .part_id = stringToInfo(part_id),
                  .bdid = stringToInfo(elements["bd_id"]),
                  .prod_id = stringToInfo(elements["prod_id"]),
                  .oper = stoi(elements["oper"])};

        *ent = entity_t{.recover_time = recover_time,
                        .outplan_time = recover_time,
                        .entity_name = elements["entity"],
                        .model_name = model,
                        .location = location,
                        .hold = false,
                        .tool = NULL,
                        .wire = NULL,
                        .job = job};
        ent->name.data.number[0] = no;
        ent->name.text_size = 4;
        ent->name.number_size = 1;

        _ents.push_back(ent);

        _entities[model][location].push_back(ent);
        _loc_ents[location].push_back(ent);
        if (_model_locations.count(model) == 0) {
            _model_locations[model] = vector<string>();
        }

        if (find(_model_locations[model].begin(), _model_locations[model].end(),
                 location) == _model_locations[model].end()) {
            _model_locations[model].push_back(location);
        }

    } else {
        perror("addMachine");
    }
}

void entities_t::addMachines(csv_t _machines, csv_t _location)
{
    int mrows = _machines.nrows();
    int lrows = _location.nrows();
    map<string, string> locations;  // entity->location
    for (int i = 0; i < lrows; ++i) {
        map<string, string> elements = _location.getElements(i);
        string ent = elements["entity"];
        string loc = elements["location"];
        locations[ent] = loc;
    }

    for (int i = 0; i < mrows; ++i) {
        map<string, string> elements = _machines.getElements(i);
        try {
            elements["location"] = locations.at(elements["entity"]);
            addMachine(elements);
        } catch (std::out_of_range &e) {
            // cout<<elements["entity"]<<endl;
            elements["log"] = "can't find the location of this entity";
            _faulty_machine.push_back(elements);
        } catch (std::invalid_argument &e) {
            elements["log"] = "information is loss";
            _faulty_machine.push_back(elements);
        }
    }

    iter(_ents, i)
    {
        _ents[i]->recover_time = ((_ents[i]->recover_time - _time) > 0
                                      ? (_ents[i]->recover_time - _time)
                                      : 0) /
                                 60.0;
        _ents[i]->job.base.end_time = _ents[i]->recover_time;
    }
}


std::map<std::string, std::map<std::string, std::vector<entity_t *> > >
entities_t::getEntities()
{
    return _entities;
}

std::map<std::string, std::vector<entity_t *> > entities_t::getLocEntity()
{
    return _loc_ents;
}

bool entityComparisonByTime(entity_t *ent1, entity_t *ent2)
{
    return ent1->recover_time < ent2->recover_time;
}

std::vector<entity_t *> entities_t::getRandomEntities(
    std::map<std::string, int> model_statistic,
    std::map<std::string, int> bdid_statistic,
    int amount)
{
    vector<entity_t *> ret;
    vector<entity_t *> pool;
    // if the model's statistic lot number isn't 0 -> push into pool
    for (std::map<std::string, int>::iterator it = model_statistic.begin();
         it != model_statistic.end(); it++) {
        if (it->second == 0)
            continue;
        for (unsigned int i = 0; i < _loc_ents[it->first].size(); ++i) {
            if (!_loc_ents[it->first][i]->hold) {
                pool.push_back(_loc_ents[it->first][i]);
            }
        }
    }

    // random_shuffle(pool.begin(), pool.end());
    sort(pool.begin(), pool.end(), entityComparisonByTime);

    if ((unsigned) amount < pool.size()) {
        ret = vector<entity_t *>(pool.begin(), pool.begin() + amount);
    } else {
        ret = vector<entity_t *>(pool.begin(), pool.end());
    }
    iter(ret, i) { ret[i]->hold = true; }
    return ret;
}

std::vector<entity_t *> entities_t::getTheSuitableEntities(
    std::map<std::string, int> model_statistic,
    std::map<std::string, int> bdid_statistic,
    int amount)
{
    vector<entity_t *> better;  // the bdid is in bdid_statistic,
    vector<entity_t *> bad;     // the bdid isn't in bdid_statistic

    vector<entity_t *> ret;
    vector<entity_t *> pool;
    // if the model's statistic lot number isn't 0 -> push into pool
    for (std::map<std::string, int>::iterator it = model_statistic.begin();
         it != model_statistic.end(); it++) {
        if (it->second == 0)
            continue;
        for (unsigned int i = 0; i < _loc_ents[it->first].size(); ++i) {
            if (!_loc_ents[it->first][i]->hold) {
                pool.push_back(_loc_ents[it->first][i]);
            }
        }
    }

    // random_shuffle(pool.begin(), pool.end());
    sort(pool.begin(), pool.end(), entityComparisonByTime);

    iter(pool, i)
    {
        string bdid = pool[i]->job.bdid.data.text;
        if (bdid_statistic.count(bdid) != 0) {
            better.push_back(pool[i]);
        } else
            bad.push_back(pool[i]);
    }

    if ((unsigned) amount < better.size()) {
        ret = vector<entity_t *>(better.begin(), better.begin() + amount);
    } else {
        ret = vector<entity_t *>(better.begin(), better.end());
    }

    iter(ret, i) { ret[i]->hold = true; }
    return ret;
}

void entities_t::reset()
{
    iter(_ents, i) { _ents[i]->hold = false; }
}

std::vector<entity_t *> entities_t::getAllEntity()
{
    return _ents;
}

std::map<std::string, std::vector<std::string> > entities_t::getModelLocation()
{
    return _model_locations;
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


vector<machine_t> entities_t::machines()
{
    vector<machine_t> ms;
    iter(_ents, i)
    {
        machine_t m = machine_t{
            .base = {.machine_no =
                         convertEntityNameToUInt(_ents[i]->entity_name),
                     .size_of_jobs = 0,
                     .available_time = _ents[i]->recover_time},
            .tool = _ents[i]->tool,
            .wire = _ents[i]->wire};
        ms.push_back(m);
    }
    return ms;
}

std::map<std::string, machine_t *> machines_t::getMachines()
{
    return _machines;
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



void machines_t::addMachines(std::vector<entity_t *> ents)
{
    iter(ents, i)
    {
        machine_t m = entityToMachine(*ents[i]);
        machine_t *m_ptr = new machine_t;
        *m_ptr = m;
        m_ptr->ptr_derived_object = ents[i];
        m_ptr->current_job.base.ptr_derived_object = &(m_ptr->current_job);
        m_ptr->base.ptr_derived_object = m_ptr;
        _machines[ents[i]->entity_name] = m_ptr;
    }
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
