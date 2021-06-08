#include <include/lot.h>
#include <pthread.h>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>
#include <set>
#include "include/machine.h"

lot_t::lot_t(std::map<std::string, std::string> elements)
{
    _route = elements["route"];
    _lot_number = elements["lot_number"];
    _pin_package = elements["pin_package"];
    _recipe = elements["bd_id"];
    _prod_id = elements["prod_id"];
    _urgent = elements["urgent_code"];

    _qty = std::stoi(elements["qty"]);
    _oper = std::stoi(elements["oper"]);
    _hold = (elements["hold"].compare("Y") == 0) ? true : false;
    _mvin = (elements["mvin"].compare("Y") == 0) ? true : false;

    _queue_time = 0;
    _fcst_time = 0;
    _outplan_time = 0;

    _finish_traversal = false;

    tmp_oper = _oper;
    tmp_mvin = _mvin;

    _is_sub_lot = _lot_number.length() >= 8 ? true : false;
    _amount_of_tools = 0;
    _amount_of_wires = 0;
}

bool lot_t::checkFormation()
{
    std::string error_msg;
    std::vector<std::string> data_members;

    if (_route.length() == 0)
        data_members.push_back("route");

    if (_lot_number.length() == 0)
        data_members.push_back("lot_number");

    if (_pin_package.length() == 0)
        data_members.push_back("pin_package");

    if (_recipe.length() == 0)
        data_members.push_back("recipe");

    if (_qty <= 0) {
        data_members.push_back("qty");
    }

    if (data_members.size()) {
        error_msg = data_members.size() > 1 ? "These information, "
                                            : "This"
                                              " information, ";
        for (unsigned int i = 0; i < data_members.size(); ++i) {
            error_msg += data_members[i] + " ,";
        }
        error_msg += data_members.size() > 1 ? " are incorrect."
                                             : " is"
                                               " incorrect.";
        addLog(error_msg);
        return false;
    }
    return true;
}

std::vector<lot_t> lot_t::createSublots()
{
    std::vector<lot_t> lots;
    if (_is_sub_lot) {
        return lots;
    }
    char str_number[100];
    int count = std::ceil((double) _qty / _lot_size);
    int remain = _qty;
    for (int i = 0; i < count; ++i) {
        sprintf(str_number, "%02d", i + 1);
        lot_t tmp(*this);
        tmp._lot_number += str_number;
        tmp._is_sub_lot = true;
        tmp.addLog("This lot is split from the parent lot " + _lot_number);
        if (remain - _lot_size > 0) {
            tmp._qty = _lot_size;
            remain -= tmp._qty;
        } else {
            tmp._qty = remain;
            remain = 0;
        }
        lots.push_back(tmp);
    }
    _qty = remain;

    return lots;
}

std::map<std::string, std::string> lot_t::data()
{
    std::map<std::string, std::string> d;
    d["route"] = _route;
    d["lot_number"] = _lot_number;
    d["pin_package"] = _pin_package;
    d["recipe"] = _recipe;
    d["prod_id"] = _prod_id;
    d["process_id"] = _process_id;
    d["bom_id"] = _bom_id;
    d["part_id"] = _part_id;
    d["part_no"] = _part_no;

    d["qty"] = std::to_string(_qty);
    d["oper"] = std::to_string(_oper);
    d["dest_oper"] = std::to_string(tmp_oper);
    d["lot_size"] = std::to_string(_lot_size);
    d["amount_of_wires"] = std::to_string(_amount_of_wires);
    d["amount_of_tools"] = std::to_string(_amount_of_tools);
    d["hold"] = _hold ? "Y" : "N";
    d["mvin"] = _mvin ? "Y" : "N";
    d["is_sub_lot"] = _is_sub_lot ? "Y" : "N";
    d["queue_time"] = std::to_string(_queue_time);
    d["fcst_time"] = std::to_string(_fcst_time);
    // d["arrival_time"] = std::to_string(
    d["log"] = join(_log, "||");

    std::vector<std::string> models;
    for (std::map<std::string, double>::iterator it = _uphs.begin();
         it != _uphs.end(); ++it) {
        models.push_back(it->first);
    }
    d["CAN_RUN_MODELS"] = join(models, ",");
    return d;
}

bool lot_t::setUph(csv_t _uph_csv){
    _uph_csv = _uph_csv.filter("recipe", _recipe);
    _uph_csv = _uph_csv.filter("oper", std::to_string(this->tmp_oper));
    double retval = true;
    if(_uph_csv.nrows() == 0){
        return false;
    } else {
        int nrows = _uph_csv.nrows();
        for(int i = 0; i < nrows; ++i){
            std::map<std::string, std::string> elements = _uph_csv.getElements(i);
            if(_uphs.count(elements["model"]) != 0){
                setUph(elements["model"], std::stof(elements["uph"]));
            }
        }
    }

    for(std::map<std::string, double>::iterator it = _uphs.begin(); it != _uphs.end(); it++){
        if(it->second == 0){
            addLog("The uph of model " + it->first + " is 0"); 
            retval = false;
        } 
    }

    return retval;
}


void lot_t::setCanRunLocation(std::map<std::string, std::vector<std::string> > model_locations){
    iter(_can_run_models, i){
        std::vector<std::string> locations = model_locations[_can_run_models[i]];
        iter(locations, j){
            if(locations[j].compare("TA-P") == 0 || locations[j].compare("TA-U") == 0) {
                if(_pin_package.find("DFN") != std::string::npos ||
                   _pin_package.find("QFN") != std::string::npos ||
                   _part_no[4] != 'A')
                {
                    continue;
                }else{
                    _can_run_locations.push_back(locations[j]);
                }
            } else if (locations[j].compare("TB-5B") == 0 || locations[j].compare("TB-5P") == 0){
                if(_pin_package.find("FBGA") != std::string::npos){
                    _can_run_locations.push_back(locations[j]);
                }else{
                    continue;
                }
            } else if (locations[j].compare("TB-P") == 0){
                if(_pin_package.find("TSOP1") != std::string::npos || _part_no[4] == 'A'){
                    continue;
                }else{
                    _can_run_locations.push_back(locations[j]);
                }
            }else{
                _can_run_locations.push_back(locations[j]);
            }
        }
    }  
}

bool lot_t::isEntityCanRun(std::string model, std::string location){
    if(_uphs.count(model) != 0){
        if(find(_can_run_locations.begin(), _can_run_locations.end(), location) != _can_run_locations.end()){
            return true;
        }
    }
    return false;
}

bool lot_t::addCanRunEntity(entity_t * ent){
    bool ret = isEntityCanRun(ent->model_name, ent->location);
    if(ret){
        _can_run_entities.push_back(ent->entity_name);
    }
    return ret;
}

job_t lot_t::job(){
    job_t j;
    memset(j.part_no.data.number, 0, sizeof(unsigned int)*8);
    j.part_no.text_size = j.part_no.number_size = 0;
    memset(j.pin_package.data.number, 0, sizeof(unsigned int)*8);
    j.pin_package.text_size = j.pin_package.number_size = 0;

    size_t lprt_no = _part_no.length();
    size_t lppkg = _pin_package.length();

    lprt_no = (lprt_no > 32 ? 32 : lprt_no);
    lppkg = (lppkg > 32 ? 32 : lppkg);

    strncpy(j.part_no.data.text, _part_no.c_str(), lprt_no);
    strncpy(j.pin_package.data.text, _pin_package.c_str(), lppkg);
        
    j.part_no.text_size = lprt_no;
    j.part_no.number_size = 32 - __builtin_clz(lprt_no >> 2) + 1;
    j.pin_package.text_size = lppkg;
    j.pin_package.number_size = 32 - __builtin_clz(lppkg >> 2) + 1;
    
    if(_urgent.length()){
        j.urgent_code = _urgent[0];
    }else
        j.urgent_code = '\0';

    return j;
}

bool lot_group_comparision(lot_group_t g1, lot_group_t g2){
    return g1.lot_amount > g2.lot_amount;
}



void lots_t::addLots(std::vector<lot_t> lots){
    this->lots = lots;
    std::string part_id, part_no;
    iter(this->lots, i){
        part_id = this->lots[i].part_id();
        part_no = this->lots[i].part_no();
        this->tool_lots[part_no].push_back(&(this->lots[i]));
        this->wire_lots[part_id].push_back(&(this->lots[i]));
        this->tool_wire_lots[part_no + "_" + part_id].push_back(&(this->lots[i]));

        amount_of_tools[this->lots[i].part_no()] = this->lots[i].getAmountOfTools();
        amount_of_wires[this->lots[i].part_id()] = this->lots[i].getAmountOfWires();
    }
}

std::map<std::string, int> lots_t::sta_models(std::map<std::string, std::map<std::string, std::vector<entity_t *> > > ents){
    std::map<std::string, int> _;
    for(std::map<std::string, std::map<std::string, std::vector<entity_t *> > >::iterator it = ents.begin(); it != ents.end(); it++){
        for(std::map<std::string, std::vector<entity_t*> >::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++){
            _[it2->first] = 0;
        } 
    }
    return _;
}

std::map<std::string, int> lots_t::sta_models(std::map<std::string, std::vector<entity_t *> > loc_ents){
    std::map<std::string, int> _;
    for(std::map<std::string, std::vector<entity_t *> >::iterator it = loc_ents.begin(); it != loc_ents.end(); it ++){
        _[it->first] = 0;    
    }
    return _;
}


std::vector<lot_group_t> lots_t::round(machines_t machines){
    
    std::map<std::string, std::map<std::string, std::vector<entity_t *> > > entities = machines.getEntities();
    std::map<std::string, std::vector<entity_t*> > loc_ents = machines.getLocEntity();
    std::map<std::string, std::vector<std::string> > model_location = machines.getModelLocation();
    
    // initialize
    iter(lots, i){
        lots[i].setCanRunLocation(model_location);
    }

    // 30 sets;
    std::vector<lot_group_t> groups;
    for(std::map<std::string, std::vector<lot_t*> >::iterator it = tool_wire_lots.begin(); it != tool_wire_lots.end(); it++){
        groups.push_back(lot_group_t{
                    .wire_tools_name = it->first,
                    .lot_amount = it->second.size()
                });
    }
    std::sort(groups.begin(), groups.end(), lot_group_comparision);

    std::vector<lot_group_t> selected_groups(groups.begin(), groups.begin() + 30);

    std::map<std::string, int> sta_tools;
    std::map<std::string, int> sta_wires;
    
    std::string t, w, t_w;
    iter(selected_groups, i){
        t_w = selected_groups[i].wire_tools_name;
        t = t_w.substr(0, t_w.find("_"));
        w = t_w.substr(t_w.find("_") + 1);
        selected_groups[i].wire_name = w;
        selected_groups[i].tool_name = t;
        if(sta_tools.count(t) == 0){
            sta_tools[t] = 0;
        }
        if(sta_wires.count(w) == 0){
            sta_wires[w] = 0;
        }

        sta_tools[t] += selected_groups[i].lot_amount;
        sta_wires[w] += selected_groups[i].lot_amount;
    }
    
    double ratio;
    iter(selected_groups, i){
        ratio = selected_groups[i].lot_amount / (double)sta_tools.at(selected_groups[i].tool_name);
        selected_groups[i].tool_amount = ratio * amount_of_tools.at(selected_groups[i].tool_name);

        ratio = selected_groups[i].lot_amount / (double)sta_wires.at(selected_groups[i].wire_name);
        selected_groups[i].wire_amount = ratio * amount_of_wires.at(selected_groups[i].wire_name);

        selected_groups[i].machine_amount = selected_groups[i].tool_amount > selected_groups[i].wire_amount ? selected_groups[i].wire_amount : selected_groups[i].tool_amount;
    }

    // models statistic
    iter(selected_groups, k){
        if(selected_groups[k].machine_amount > 0){
            t_w = selected_groups[k].wire_tools_name;
            std::vector<lot_t*> lots = this->tool_wire_lots[t_w];
            // std::map<std::string, int> sta_models;
             
            std::map<std::string, int>  sta = sta_models(loc_ents);
            for(unsigned int i = 0; i < lots.size(); ++i){
                std::vector<std::string> can_run_locations = lots[i]->can_run_locations();
                iter(can_run_locations, c){
                    sta[can_run_locations[c]] += 1;
                }
            }
            for(std::map<std::string, int>::iterator it = sta.begin(); it != sta.end(); ++it){
                printf("%s : %d\n", it->first.c_str(), it->second);
            } 
            printf("***********************************\n");
            selected_groups[k].models_statistic = sta;
        }else{
            printf("Error!\n");
        }
    } 
    iter(selected_groups, i){
        selected_groups[i].entities = machines.randomlyGetEntitiesByLocations(selected_groups[i].models_statistic, selected_groups[i].machine_amount);
    }

    // check 
    std::set<entity_t *> s;
    iter(selected_groups, i){
        iter(selected_groups[i].entities, j){
            if(s.count(selected_groups[i].entities[j]) == 0){
                s.insert(selected_groups[i].entities[j]);
            }else{
                printf("group %d is duplicated!\n", i);
            }
        }
        if(selected_groups[i].entities.size() < selected_groups[i].machine_amount){
            printf("goup %d is sufficient of machines\n", i);
        }

        std::string tool_wire_name = selected_groups[i].wire_tools_name;
        std::vector<lot_t*> lots = tool_wire_lots[tool_wire_name];
        std::vector<lot_t*> failed;
        std::vector<lot_t*> successful;
        bool found = false;
        iter(lots, j){
            iter(selected_groups[i].entities, k){
                if(lots[j]->addCanRunEntity(selected_groups[i].entities[k])){
                    found = true;
                }            
            }
            if(found)
                successful.push_back(lots[j]);
            else
                failed.push_back(lots[j]);
        }
        tool_wire_lots[tool_wire_name] = failed;
        selected_groups[i].lots = successful;
    }
    return selected_groups;
}


