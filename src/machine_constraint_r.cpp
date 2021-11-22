#include "include/machine_constraint_r.h"
#include "include/machine_constraint.h"

#include <regex>
#include <string>
#include <vector>

using namespace std;

machine_constraint_r_t::machine_constraint_r_t(csv_t csv)
    : machine_constraint_t(csv)
{
}

bool machine_constraint_r_t::_isMachineRestrained(std::regex entity_re,
                                                  std::string restrained_model,
                                                  std::string entity_name,
                                                  std::string machine_model)
{
    // R means reject if entity_re matches and two given models are the same
    // if one of the condition isn't matched, the machine is okay
    return !regex_match(entity_name, entity_re) ||
           restrained_model != machine_model;
}
