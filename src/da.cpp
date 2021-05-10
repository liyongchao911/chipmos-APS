#include <include/da.h>
#include <stdexcept>

da_stations_t::da_stations_t(csv_t fcst, bool strict){
    setFcst(fcst, strict);
}

int da_stations_t::setFcst(csv_t _fcst, bool strict){
    int nrows = _fcst.nrows();
    std::map<std::string, std::string> elements;
    std::string bd_id;
    double fcst, act, remain;
    bool retval = true;
    for(int i = 0; i < nrows; ++i){
        elements = _fcst.getElements(i);
        bd_id = elements["bd_id"]; 
        if(bd_id.length() && bd_id.compare("(null)") != 0){ // remove empty bd_id
            if(_da_stations_container.count(bd_id) != 0){
                if(strict){
                    std::string error_msg = "bd_id :" + bd_id + " is duplicated";
                    throw std::invalid_argument(error_msg);
                }else{
                    retval = false;
                }
            }
            // overwrite
            fcst = std::stod(elements["da_out"]) * 1000;
            
            if(fcst == 0)
                continue;

            act = std::stod(elements["da_act"]) * 1000;
            remain = fcst; // TODO : remain = fcst - act may less than 0
             
            _da_stations_container[bd_id] = da_station_t{
                .fcst = fcst,
                .act = act,
                .remain = remain,
                .upm = fcst / 1440
            };
             
        }
    }
    return retval;
}

bool da_stations_t::addArrivedLotToDA(lot_t &lot){
    try{
        _da_stations_container.at(lot.recipe()).arrived.push_back(lot);
    }catch(std::out_of_range & e){
        std::string error_msg;
        error_msg += std::string("In function da_stations_t::addArrivedLotToDA trigger exception : ") +  e.what() + " fcst doesn't have " + lot.recipe() + " this recipe, "
            "but lot_number " + lot.lotNumber() + " does.";
        throw(std::out_of_range(error_msg));
    }
    return true;
}

bool da_stations_t::addUnarrivedLotToDA(lot_t &lot){
    try{
        _da_stations_container.at(lot.recipe()).unarrived.push_back(lot);
    }catch(std::out_of_range & e){
        std::string error_msg;
        error_msg += std::string("In function da_stations_t::addUnarrivedLotToDA trigger exception : ") + e.what() + " fcst doesn't have " + lot.recipe() + " this recipe"
            "but lot_number " + lot.lotNumber() + " does";
        throw(std::out_of_range(error_msg));
    }
    return true;
}


std::vector<lot_t > da_stations_t::distributeProductionCapacity(){
    std::vector<lot_t> lots;
    // arrived first
    //
    return lots;
}
