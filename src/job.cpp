#include <include/job.h>

lot_t::lot_t(std::map<std::string, std::string> elements){
    _route = elements["route"];
    _lot_number = elements["lot_number"];
    _pin_package = elements["pin_package"];
    
    _qty = std::stoi(elements["qty"]);
    _oper = std::stoi(elements["oper"]);
    _hold = (elements["hold"].compare("Y") == 0) ? true: false;
    _mvin = (elements["mvin"].compare("Y") == 0) ? true: false;

}

int lot_t::oper(){
    return _oper;
}

std::string lot_t::route(){
    return _route;
}
