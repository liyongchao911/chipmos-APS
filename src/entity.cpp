#include <include/entity.h>
#include <limits.h>
#include <algorithm>
#include <iterator>
#include <stdexcept>

using namespace std;

entities_t::entities_t(string _time){
    time = 0;
    min_outplan_time = LONG_LONG_MAX;
    setTime(_time); 
}

void entities_t::setTime(string _time){
    if(_time.length()){
        time = timeConverter(_time);
    }
}


void entities_t::addMachine(map<string, string> elements){
    if(elements["recover_time"].length() == 0){
        throw std::invalid_argument("recover time is empty");
    }
    double recover_time = timeConverter(elements["recover_time"]);

    
    entity_t * ent = new entity_t();
    string model, location;
    model = elements["model"];
    location = elements["location"];
    if(ent){
        *ent = entity_t{
            .recover_time = recover_time,
            .outplan_time = recover_time,
            .entity_name = elements["entity"],
            .model_name = model,
            .location = location,
            .hold = false
        };
        if(ent->recover_time < min_outplan_time)
            min_outplan_time = ent->recover_time;

        ents.push_back(ent);
        
        _entities[model][location].push_back(ent);
        loc_ents[location].push_back(ent);
        if(model_locations.count(model) == 0){
            model_locations[model] = vector<string>();
        } 

        if(find(model_locations[model].begin(), model_locations[model].end(), location) == model_locations[model].end()){
            model_locations[model].push_back(location); 
        }
         
    }else {
        perror("addMachine");
    }
}

void entities_t::addMachines(csv_t _machines, csv_t _location){
    int mrows = _machines.nrows();
    int lrows = _location.nrows();
    map<string, string> locations; // entity->location 
    for(int i = 0; i < lrows; ++i){
        map<string, string> elements = _location.getElements(i);
        string ent = elements["entity"];
        string loc = elements["location"];
        locations[ent] = loc;
    }

    for(int i = 0; i < mrows; ++i){
        map<string, string> elements = _machines.getElements(i);
        try{
            elements["location"] = locations.at(elements["entity"]);
            addMachine(elements);
        }catch(std::out_of_range & e){
            // cout<<elements["entity"]<<endl;
            elements["log"] = "can't find the location of this entity";
            faulty_machine.push_back(elements);
        }catch(std::invalid_argument & e){
            elements["log"] = "information is loss";
            faulty_machine.push_back(elements);
        }
    }
    
    iter(ents, i){
        ents[i]->recover_time = (ents[i]->recover_time - min_outplan_time) / 60.0;
    }
}


std::map<std::string, std::map<std::string, std::vector<entity_t *> > >
entities_t::getEntities(){
    return _entities;
}

std::map<std::string, std::vector<entity_t *> > entities_t::getLocEntity(){
    return loc_ents;
}

std::vector<entity_t *> entities_t::randomlyGetEntitiesByLocations(std::map<std::string, int> statistic, int amount){
    vector<entity_t *> ret;

    vector<entity_t *> pool;
    for(std::map<std::string, int>::iterator it = statistic.begin(); it != statistic.end(); it++){
        if(it->second == 0)
            continue;
        for(unsigned int i = 0; i < loc_ents[it->first].size(); ++i){
            if(!loc_ents[it->first][i]->hold){
                pool.push_back(loc_ents[it->first][i]);
            }
        }
    }

    random_shuffle(pool.begin(), pool.end());
    
    if((unsigned)amount < pool.size()){
        ret = vector<entity_t*>(pool.begin(), pool.begin() + amount); 
    }else{
        ret = vector<entity_t*>(pool.begin(), pool.end());
    }

    iter(ret, i){
        ret[i]->hold = true;
    }
    return ret;
}

void entities_t::reset(){
    iter(ents, i){
        ents[i]->hold = false;
    }
}

std::map<std::string, std::vector<std::string> > entities_t::getModelLocation(){
    return model_locations;
}
