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

    checkFormation();
}

// lot_t::lot_t(lot_t & lot){
//     this->_route = lot._route;
//     this->_lot_number = lot._lot_number;
//     this->_pin_package = lot._pin_package;
//     this->_recipe = lot._recipe;
//     this->_prod_id = lot._prod_id;
//     this->_process_id = lot._process_id;
//     this->_qty = lot._qty;
//     this->_oper = lot._oper;
//     this->_lot_size = lot._lot_size;
//     this->_hold = lot._hold;
//     this->_mvin = lot._mvin;
//     this->_queue_time = lot._queue_time;
//     this->_fcst_time = lot._fcst_time;
//     this->_outplan_time = lot._outplan_time;
//     this->_finish_traversal = lot._finish_traversal;
//     this->tmp_mvin = lot.tmp_mvin;
//     this->tmp_oper = lot.tmp_oper;
// }


void lot_t::checkFormation()
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

    if (data_members.size()) {
        error_msg = data_members.size() > 1 ? "These"
                                            : "This"
                                              " information, ";
        for (unsigned int i = 0; i < data_members.size(); ++i) {
            error_msg += data_members[i];
        }
        error_msg += data_members.size() > 1 ? ", are"
                                             : ", is"
                                               " not provided";

        throw std::invalid_argument(error_msg);
    }
}

std::vector<lot_t> lot_t::createSublots()
{
    std::vector<lot_t> lots;
    if (_lot_number.length() >= 8) {
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
        tmp._log.clear();
        if (remain - _lot_size > 0) {
            tmp._qty = _lot_size;
            remain -= tmp._qty;
        } else {
            tmp._qty = remain;
            remain = 0;
        }
        lots.push_back(tmp);
    }

    return lots;
}
