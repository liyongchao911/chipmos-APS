//
// Created by YuChunLin on 2021/11/22.
//


#include "include/machine_constraint_a.h"
#include "include/machine_constraint.h"

#include <regex>
#include <string>
#include <vector>

using namespace std;

machine_constraint_a_t::machine_constraint_a_t(csv_t csv)
    : machine_constraint_t(csv)
{
}

bool machine_constraint_a_t::_isMachineRestrained(std::regex entity_re,
                                                  std::string restrained_model,
                                                  std::string entity_name,
                                                  std::string machine_model)
{
    // a means acceptance if entity_re matches and two given models are the same
    return regex_match(entity_name, entity_re) &&
           restrained_model == machine_model;
}
