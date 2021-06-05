#include <include/machine.h>
#include <stdexcept>

using namespace std;

machines_t::machines_t(const char *_time){
    time = timeConverter(_time);
}

void machines_t::addMachine(map<string, string> elements){
    if(elements["recover_time"].length() == 0){
        throw std::invalid_argument("recover time is empty");
    }
    double recover_time = timeConverter(elements["recover_time"]);
    
    recover_time -= time;
     
    _entities[elements["model"]][elements["location"]].push_back(entity_t{
                .recover_time = recover_time,
                .entity_name = elements["entity"],
                .model_name = elements["model"],
                .location = elements["location"]
            });
}

void machines_t::addMachines(csv_t _machines, csv_t _location){
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
    
}
