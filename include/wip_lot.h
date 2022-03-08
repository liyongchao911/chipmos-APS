#ifndef __LOT_WIP_H__
#define __LOT_WIP_H__

#include <string>
#include "include/lot_base.h"

class lot_wip_t : public lot_base_t
{
protected:
    std::string _route;
    std::string _urgent_code;

    std::string _last_wb_entity;
    int _sublot_size;

    virtual inline std::map<std::string, std::vector<std::string> >
    getRowDefaultValues() const override;

public:
    lot_wip_t();
    lot_wip_t(std::map<std::string, std::string> &row);
};

std::map<std::string, std::vector<std::string> >
lot_wip_t::getRowDefaultValues() const
{
    std::map<std::string, std::vector<std::string> > default_values =
        lot_base_t::getRowDefaultValues();

    default_values[""].push_back("route");
    default_values[""].push_back("urgent_code");
    default_values[""].push_back("last_wb_entity");
    default_values["1"].push_back("sub_lot_size");

    return default_values;
}

#endif
