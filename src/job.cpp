#include <include/job.h>
#include <cmath>
#include <stdexcept>

lot_t::lot_t(std::map<std::string, std::string> elements)
{
    _route = elements["route"];
    _lot_number = elements["lot_number"];
    _pin_package = elements["pin_package"];
    _recipe = elements["bd_id"];
    _prod_id = elements["prod_id"];

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
