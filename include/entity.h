#ifndef __ENTITY_H__
#define __ENTITY_H__

#include <include/csv.h>
#include <time.h>
#include <map>
#include <string>
#include <vector>

typedef struct {
    double recover_time;
    double outplan_time;
    std::string entity_name;
    std::string model_name;
    std::string location;
    bool hold;
} entity_t;


class entities_t
{
private:

    std::vector<entity_t *> ents;
    
    // _entities[MODEL][AREA] is a vector of entity_t object.
    std::map<std::string, std::map<std::string, std::vector<entity_t *> > >
        _entities;

    std::map<std::string, std::vector<std::string> > model_locations;

    std::map<std::string, std::vector<entity_t *> > loc_ents;

    std::vector<std::map<std::string, std::string> > faulty_machine;


    time_t time;
    time_t min_outplan_time;

public:
    /**
     * entities_t () - constructor of entities_t
     *
     * The constructor will convert @b _time to time_t type
     */
    entities_t(std::string _time);
    
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
    void addMachines(csv_t machines_csv, csv_t location_csv);

    // /**
    //  * randomlyGetEntities () - randomly get the entities by model and area
    //  */
    // std::vector<entity_t> randomlyGetEntities(std::string model_name,
    //                                           std::string area,
    //                                           int amount);
    
    /**
     * randomlyGetEntitiesByLocations
     */
    std::vector<entity_t *> randomlyGetEntitiesByLocations(std::map<std::string, int> statistic, int amount);

    void setTime(std::string text);
    
    std::map<std::string, std::map<std::string, std::vector<entity_t *> > > getEntities();

    std::map<std::string, std::vector<entity_t *> > getLocEntity();

    std::map<std::string, std::vector<std::string> > getModelLocation();

    void reset();
};

#endif
