#ifndef __ENTITY_H__
#define __ENTITY_H__

#include <ctime>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/csv.h"
#include "include/lot.h"
#include "include/machine.h"

unsigned int convertEntityNameToUInt(std::string name);
std::string convertUIntToEntityName(unsigned int);

class entity_t;
machine_t entityToMachine(entity_t ent);

class entity_t
{
private:
    double _recover_time;
    double _outplan_time;
    std::string _entity_name;
    std::string _model_name;
    std::string _location;

    lot_t _current_lot;

    std::vector<lot_t *> _prescheduled_lots;

public:
    entity_t() {}

    entity_t(std::map<std::string, std::string> elements);

    double getRecoverTime() const;
    double getOutplanTime() const;
    std::string getEntityName();
    std::string getModelName();
    std::string getLocation();

    void addPrescheduledLot(lot_t *lot);

    void prescheduleLots();

    virtual machine_t machine();

    bool hold;
};

inline double entity_t::getRecoverTime() const
{
    return _recover_time;
}

inline double entity_t::getOutplanTime() const
{
    return _outplan_time;
}

inline std::string entity_t::getEntityName()
{
    return _entity_name;
}

inline std::string entity_t::getModelName()
{
    return _model_name;
}

inline std::string entity_t::getLocation()
{
    return _location;
}

inline void entity_t::addPrescheduledLot(lot_t *lot)
{
    if (lot->isPrescheduled())
        this->_prescheduled_lots.push_back(lot);
    else
        throw std::invalid_argument("lot is not prescheduled lot");
}



class ancillary_resources_t
{
private:
    std::map<std::string, std::vector<ares_t *> > _tools;

public:
    ancillary_resources_t(std::map<std::string, int> tools);
    std::vector<tool_t *> aRound(std::map<std::string, int> amount);
    std::vector<tool_t *> aRound(std::string, int);
};

class machines_t
{
private:
    std::map<std::string, machine_t *> _machines;

public:
    void addMachines(std::vector<entity_t *> ents);

    std::map<std::string, machine_t *> getMachines();
};


#endif
