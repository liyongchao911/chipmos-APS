#ifndef __MACHINE_H__
#define __MACHINE_H__

#include <include/csv.h>
#include <time.h>
#include <map>
#include <string>
#include <vector>

typedef struct {
    double recover_time;
    std::string entity_name;
    std::string model_name;
    std::string location;
} entity_t;


class machines_t
{
private:
    // _entities[MODEL][AREA] is a vector of entity_t object.
    std::map<std::string, std::map<std::string, std::vector<entity_t> > >
        _entities;

    std::vector<std::map<std::string, std::string> > faulty_machine;
    time_t time;

public:
    /**
     * machines_t () - constructor of machines_t
     *
     * The constructor will convert @b _time to time_t type
     */
    machines_t(char *_time);

    /**
     * addMachine() - add a machine
     *
     * @b elements is a std::map container which store the relationship between
     * header and data. For example, elements[ENTITY] == "BB211",
     * elements[MODEL] = "UTC3000" and etc... This function will convert
     * elements to entity_t object.
     */
    void addMachine(std::map<std::string, std::string> elements);

    /**
     * addMachines() - add machines from dataframe
     *
     * add machines from @csv_t type dataframe.
     */
    void addMachines(csv_t machines, csv_t location);

    /**
     * randomlyGetEntities () - randomly get the entities by model and area
     */
    std::vector<entity_t> randomlyGetEntities(std::string model_name,
                                              std::string area,
                                              int amount);
};

#endif
