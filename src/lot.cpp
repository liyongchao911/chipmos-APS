#include <include/lot.h>
#include <pthread.h>
#include <cmath>
#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include "include/entity.h"
#include "include/job.h"
#include "include/job_base.h"

lot_t::lot_t(std::map<std::string, std::string> elements)
{
    _route = elements["route"];
    _lot_number = elements["lot_number"];
    _pin_package = elements["pin_package"];
    _recipe = elements["bd_id"];
    _prod_id = elements["prod_id"];
    _urgent = elements["urgent_code"];
    _customer = elements["customer"];

    _qty = std::stoi(elements["qty"]);
    _oper = std::stoi(elements["oper"]);
    _hold = (elements["hold"].compare("Y") == 0) ? true : false;
    _mvin = (elements["mvin"].compare("Y") == 0) ? true : false;
    
    if(elements.count("queue_time") == 0){
        _queue_time = 0;
    }else{
        _queue_time = std::stod(elements["queue_time"]);
    }
    
    _queue_time = (elements.count("queue_time") == 0 ? 0 : std::stod(elements["queue_time"]));

    _fcst_time =  (elements.count("fcst_time") == 0 ? 0 : std::stod(elements["fcst_time"]));
    _outplan_time = 0;

    _finish_traversal = false;

    tmp_oper = _oper;
    tmp_mvin = _mvin;

    _is_sub_lot = _lot_number.length() >= 8 ? true : false;
    _amount_of_tools = elements.count("amount_of_tools") == 0 ? 0 : std::stoi(elements["amount_of_tools"]);
    _amount_of_wires = elements.count("amount_of_wires") == 0 ? 0 : std::stoi(elements["amount_of_wires"]);

    if(elements.count("CAN_RUN_MODELS") != 0 && elements.count("PROCESS_TIME") != 0 && elements.count("uphs")){
        char * text = strdup(elements["CAN_RUN_MODELS"].c_str());
        std::vector<std::string> models = split(text, ',');
        free(text);
        text = strdup(elements["PROCESS_TIME"].c_str());
        std::vector<std::string> ptimes = split(text, ',');
        free(text);
        text = strdup(elements["uphs"].c_str());
        std::vector<std::string> uphs = split(text, ','); 
        free(text);
        _can_run_models = models;
        if(models.size() != ptimes.size()){
            throw std::invalid_argument("vector size is not the same, lot_number : " + _lot_number);
        }else{
            iter(models, i){
                _model_process_times[models[i]] = std::stod(ptimes[i]);
                _uphs[models[i]] = std::stod(uphs[i]);
            }
        }
    }

    _part_id = elements.count("part_id") == 0 ? "" : elements["part_id"];
    _part_no = elements.count("part_no") == 0 ? "" : elements["part_no"]; 
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
    d["log"] = join(_log, "||");
    d["urgent_code"] = _urgent;
    d["customer"] = _customer;

    std::vector<std::string> models;
    for (std::map<std::string, double>::iterator it = _uphs.begin();
         it != _uphs.end(); ++it) {
        models.push_back(it->first);
    }
    
    std::vector<std::string> process_times;
    for(std::map<std::string, double>::iterator it = _model_process_times.begin(); it != _model_process_times.end(); it++){
        process_times.push_back(std::to_string(it->second));
    }
    
    std::vector<std::string> uphs;
    for(std::map<std::string, double>::iterator it = _uphs.begin(); it!= _uphs.end(); it++){
        uphs.push_back(std::to_string(it->second));
    }
    
    d["CAN_RUN_MODELS"] = join(models, ",");
    d["PROCESS_TIME"] = join(process_times, ",");
    d["uphs"] = join(uphs, ",");
    return d;
}

bool lot_t::setUph(csv_t _uph_csv){
    _uph_csv = _uph_csv.filter("recipe", _recipe);
    _uph_csv = _uph_csv.filter("oper", std::to_string(this->tmp_oper));
    _uph_csv = _uph_csv.filter("cust", _customer);
    if(_uph_csv.nrows() == 0){
        this->addLog("(" + std::to_string(this->tmp_oper) + ", "+ _recipe + ") is not in uph file");
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
    std::vector<std::string> invalid_models; 
    for(std::map<std::string, double>::iterator it = _uphs.begin(); it != _uphs.end(); it++){
        if(it->second == 0){
            invalid_models.push_back(it->first);
        }
    }
    iter(invalid_models, i){
        _uphs.erase(invalid_models[i]);
        _model_process_times.erase(invalid_models[i]);
    }

    if(_uphs.size() == 0){
        addLog("All of uph is 0");
        return false;
    }else
        return true;
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
        _entity_process_times[ent->entity_name] = _model_process_times[ent->model_name];
    }
    return ret;
}



job_t lot_t::job(){
    job_t j;
    job_base_init(&j.base); 
    _list_init(&j.list);
    j.list.get_value = jobGetValue;

    j.part_no = stringToInfo(_part_no);
    j.pin_package = stringToInfo(_pin_package);
    j.base.job_info = stringToInfo(_lot_number);
    j.customer = stringToInfo(_customer);
    j.part_id = stringToInfo(_part_id);
    j.bdid = stringToInfo(_recipe);


    if(_urgent.length()){
        j.urgent_code = _urgent[0];
    }else
        j.urgent_code = '\0';

    j.base.qty = _qty;
    j.base.start_time = j.base.end_time = 0;
    j.base.arriv_t = _queue_time;
    
    return j;
}

std::map<std::string, double> lot_t::getEntitiyProcessTime(){
    return _entity_process_times;
}

bool lot_group_comparision(lot_group_t g1, lot_group_t g2){
    return g1.lot_amount > g2.lot_amount;
}



