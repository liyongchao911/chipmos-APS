#include <include/job.h>
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

    checkFormation();
}

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
