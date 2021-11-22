#include "include/machine_constraint.h"
#include <algorithm>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>
#include "include/job.h"

using namespace std;

string machine_constraint_t::transformStarToRegexString(std::string str)
{
    regex star("\\*");
    return regex_replace(str, star, R"(\w*)"s);
}

string machine_constraint_t::transformPkgIdToRegex(string pkg_id)
{
    regex left_parenthese("\\{");
    regex right_parenthese("\\}");
    pkg_id = transformStarToRegexString(pkg_id);
    pkg_id = regex_replace(pkg_id, left_parenthese, R"(\{)");
    pkg_id = regex_replace(pkg_id, right_parenthese, R"(\})");
    return pkg_id;
}

string machine_constraint_t::transformEntityGroupToRegex(string entity_group)
{
    return transformStarToRegexString(entity_group);
}

void machine_constraint_t::_storeStationData(csv_t csv, int oper)
{
    constraint_oper_t station_data;

    if (csv.nrows()) {
        csv_t restrained_entity = csv.filter("el_key", "RESTRAINED-ENTITY");
        _storeRestrainedData(restrained_entity, station_data.restrained_entity);

        csv_t restrained_pinpkg = csv.filter("el_key", "RESTRAINED-ENT-PINPK");
        _storeRestrainedData(restrained_pinpkg,
                             station_data.restrained_entity_pinpkg);
        _table[oper] = station_data;
    }
}

void machine_constraint_t::_storeRestrainedData(
    csv_t csv,
    map<string, vector<constraint_entry_t>> &restrained_ent)
{
    for (int i = 0, size = csv.nrows(); i < size; ++i) {
        map<string, string> row = csv.getElements(i);
        string cust = row["el_customer"];
        string pkg_id = transformPkgIdToRegex(row["el_check_value"]);
        string entity_group =
            transformEntityGroupToRegex(row["el_entity_group"]);

        if (restrained_ent.count(cust) == 0) {
            restrained_ent[cust] = vector<constraint_entry_t>();
            // cout << cust << endl;
        }
        restrained_ent[cust].push_back(
            constraint_entry_t{regex(pkg_id), pkg_id, regex(entity_group),
                               entity_group, row["el_entity_model"]});
    }
}



machine_constraint_t::machine_constraint_t(csv_t csv)
{
    for (int i = 0; i < NUMBER_OF_WB_STATIONS; ++i) {
        csv_t tmp = csv.filter("el_entity_oper", to_string(WB_STATIONS[i]));
        _storeStationData(tmp, WB_STATIONS[i]);
    }
}

vector<machine_t *> machine_constraint_t::getAvailableMachines(
    job_t *job,
    vector<machine_t *> machines)
{
    string pkg_id(job->pkg_id.data.text);
    string pin_pkg(job->pin_package.data.text);
    string cust(job->customer.data.text);
    return this->_getAvailableMachines(pin_pkg, pkg_id, cust, job->oper,
                                       machines);
}


std::vector<machine_t *> machine_constraint_t::_getRestrainedMachine(
    const std::string &val,
    std::vector<constraint_entry_t> entries,
    std::vector<machine_t *> machine_group)
{
    vector<machine_t *> result;
    foreach (entries, i) {
        regex _entry_check_val = entries[i].check_val;
        if (regex_match(val, _entry_check_val)) {
            regex entity_re = entries[i].entity_name;
            string restrained_model = entries[i].model;
            foreach (machine_group, j) {
                string machine_name(
                    machine_group[j]->base.machine_no.data.text);
                string machine_model(machine_group[j]->model_name.data.text);
                if (_isMachineRestrained(entity_re, restrained_model,
                                         machine_name, machine_model)) {
                    result.push_back(machine_group[j]);
                }
            }
        }
    }
    return result;
}

vector<machine_t *> machine_constraint_t::_getAvailableMachines(
    std::string pin_pkg,
    std::string pkg_id,
    std::string customer,
    int oper,
    std::vector<machine_t *> machines)
{
    vector<machine_t *> group;

    if (_table.count(oper) == 0)
        return machines;
    else {
        constraint_oper_t station_table = _table.at(oper);
        // restrained entity pinpkg
        if (station_table.restrained_entity_pinpkg.count(customer) != 0) {
            group = _getRestrainedMachine(
                pin_pkg, station_table.restrained_entity_pinpkg.at(customer),
                machines);
        }

        // restrained entity
        if (station_table.restrained_entity.count(customer) != 0) {
            group += _getRestrainedMachine(
                pkg_id, station_table.restrained_entity.at(customer), machines);
        }
    }

    set<machine_t *> machine_set(group.begin(), group.end());
    return vector<machine_t *>(machine_set.begin(), machine_set.end());
}