#include "include/wip_lot.h"

using namespace std;

lot_wip_t::lot_wip_t()
    : lot_base_t(), _route(), _urgent_code(), _last_wb_entity(), _sublot_size(1)
{
}

lot_wip_t::lot_wip_t(std::map<std::string, std::string> &row)
    : lot_base_t(row), _route()
{
    _setupDefaultValueOfRow(row);

    _route = row.at("route");
    _urgent_code = row.at("urgent_code");
    _last_wb_entity = row.at("last_wb_entity");
    _sublot_size = stoi(row.at("sub_lot_size"));
}
