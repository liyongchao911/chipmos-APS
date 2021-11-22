#ifndef __MACHINE_CONSTRAINT__
#define __MACHINE_CONSTRAINT__

#include "include/csv.h"
#include "include/job.h"
#include "include/machine.h"
#include "include/route.h"

#include <map>
#include <regex>
#include <string>
#include <vector>



struct constraint_entry_t {
    std::regex check_val;
    std::string _str_check_val;
    std::regex entity_name;
    std::string _str_entity_name;
    std::string model;
};

struct constraint_oper_t {
    std::map<std::string, std::vector<constraint_entry_t>> restrained_entity;
    std::map<std::string, std::vector<constraint_entry_t>>
        restrained_entity_pinpkg;

    constraint_oper_t() : restrained_entity(), restrained_entity_pinpkg() {}
};

class machine_constraint_t
{
private:
    void _storeStationData(csv_t csv, int oper);
    void _storeRestrainedData(
        csv_t csv,
        std::map<std::string, std::vector<constraint_entry_t>> &restraint_ent);

protected:
    std::map<int, constraint_oper_t> _table;

    std::vector<machine_t *> _getRestrainedMachine(
        const std::string &val,
        std::vector<constraint_entry_t> entries,
        std::vector<machine_t *> machine_group);
    virtual std::vector<machine_t *> _getAvailableMachines(
        std::string pin_pkg,
        std::string pkg_id,
        std::string customer,
        int oper,
        std::vector<machine_t *> machines);
    virtual bool _isMachineRestrained(std::regex entity_re,
                                      std::string restrained_model,
                                      std::string entity_name,
                                      std::string model_name) = 0;

    machine_constraint_t(){};

public:
    machine_constraint_t(csv_t csv);
    static std::string transformStarToRegexString(std::string str);
    static std::string transformPkgIdToRegex(std::string pkg_id);
    static std::string transformEntityGroupToRegex(std::string entity_group);

    std::vector<machine_t *> getAvailableMachines(job_t *job,
                                                  std::vector<machine_t *>);
};


#endif
