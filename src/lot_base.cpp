#include "include/lot_base.h"
#include "include/linked_list.h"

using namespace std;

map<string, vector<string> > lot_base_t::getRowDefaultValues() const
{
    return {{"",
             {"lot_number", "pin_package", "recipe", "prod_id", "part_id",
              "part_no", "pkg_id", "customer"}},
            {"0", {"qty", "oper"}}};
}

void lot_base_t::_setupDefaultValueOfRow(map<string, string> &row)
{
    map<string, vector<string> > default_values = getRowDefaultValues();
    for (auto it = default_values.begin(); it != default_values.end(); ++it) {
        for (unsigned int i = 0; i < it->second.size(); ++i) {
            if (row.count(it->second[i]) == 0)
                row[it->second[i]] = it->first;
        }
    }
}


lot_base_t::lot_base_t()
    : job_base_t(),
      list_ele_t(),
      _lot_number(),
      _pin_package(),
      _recipe(),
      _prod_id(),
      _part_id(),
      _part_no(),
      _pkg_id(),
      _customer(),
      _log(),
      _can_run_models(),
      _can_run_locations()
{
    _oper = _qty = 0;
    _number_of_tools = _number_of_wires = 0;
    _is_sub_lot = _is_automotive = _spr_hot = _hold = _mvin = false;
    _cr = 0.0;
}

lot_base_t::lot_base_t(std::map<std::string, std::string> &row)
    : job_base_t(),
      list_ele_t(),
      _lot_number(),
      _pin_package(),
      _recipe(),
      _prod_id(),
      _part_id(),
      _part_no(),
      _pkg_id(),
      _customer(),
      _log(),
      _can_run_models(),
      _can_run_locations()
{
    _setupDefaultValueOfRow(row);

    _lot_number = row.at("lot_number");
    _pin_package = row.at("pin_package");
    _recipe = row.at("recipe");
    _prod_id = row.at("prod_id");
    _part_id = row.at("part_id");
    _part_no = row.at("part_no");
    _pkg_id = row.at("pkg_id");
    _customer = row.at("customer");
    _oper = stoi(row.at("oper"));
    _qty = stoi(row.at("qty"));
    _number_of_tools = _number_of_wires = 0;
    _is_sub_lot = _is_automotive = _spr_hot = _hold = _mvin = false;
    _cr = 0.0;
}
