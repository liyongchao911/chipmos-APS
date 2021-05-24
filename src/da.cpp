#include <include/common.h>
#include <include/da.h>
#include <stdexcept>

da_stations_t::da_stations_t(csv_t fcst, bool strict)
{
    setFcst(fcst, strict);
}

int da_stations_t::setFcst(csv_t _fcst, bool strict)
{
    int nrows = _fcst.nrows();
    std::map<std::string, std::string> elements;
    std::string bd_id;
    double fcst, act, remain;
    bool retval = true;
    for (int i = 0; i < nrows; ++i) {
        elements = _fcst.getElements(i);
        bd_id = elements["bd_id"];
        if (bd_id.length() &&
            bd_id.compare("(null)") != 0) {  // remove empty bd_id
            if (_da_stations_container.count(bd_id) != 0) {
                if (strict) {
                    std::string error_msg =
                        "bd_id :" + bd_id + " is duplicated";
                    throw std::invalid_argument(error_msg);
                } else {
                    retval = false;
                }
            }
            // overwrite
            fcst = std::stod(elements["da_out"]) * 1000;

            if (fcst == 0)
                continue;

            act = std::stod(elements["da_act"]) * 1000;
            remain = fcst;  // FIXME : remain = fcst - act may less than 0

            _da_stations_container[bd_id] = da_station_t{.fcst = fcst,
                                                         .act = act,
                                                         .remain = remain,
                                                         .upm = fcst / 1440,
                                                         .time = 0,
                                                         .finished = false};
        }
    }
    return retval;
}

bool da_stations_t::addArrivedLotToDA(lot_t &lot)
{
    try {
        _da_stations_container.at(lot.recipe()).arrived.push_back(lot);
    } catch (std::out_of_range &e) {
        std::string error_msg;
        error_msg += std::string(
                         "In function da_stations_t::addArrivedLotToDA trigger "
                         "exception : ") +
                     e.what() + " fcst doesn't have " + lot.recipe() +
                     " this recipe, "
                     "but lot_number " +
                     lot.lotNumber() + " does.";
        throw(std::out_of_range(error_msg));
    }
    return true;
}

bool da_stations_t::addUnarrivedLotToDA(lot_t &lot)
{
    try {
        _da_stations_container.at(lot.recipe()).unarrived.push_back(lot);
    } catch (std::out_of_range &e) {
        std::string error_msg;
        error_msg += std::string(
                         "In function da_stations_t::addUnarrivedLotToDA "
                         "trigger exception : ") +
                     e.what() + " fcst doesn't have " + lot.recipe() +
                     " this recipe"
                     "but lot_number " +
                     lot.lotNumber() + " does";
        throw(std::out_of_range(error_msg));
    }
    return true;
}


std::vector<lot_t> da_stations_t::distributeProductionCapacity()
{
    std::vector<lot_t> lots;
    // arrived first

    // TODO: sorting
    for (std::map<std::string, da_station_t>::iterator it =
             _da_stations_container.begin();
         it != _da_stations_container.end(); it++) {
        lots += daDistributeCapacity(it->second);
    }

    return lots;
}


std::vector<lot_t> da_stations_t::daDistributeCapacity(da_station_t &da)
{
    // lot -> sublots
    std::vector<lot_t> arrived_lots;
    std::vector<lot_t> unarrived_lots;
    std::vector<lot_t> result;

    arrived_lots = getSubLot(da.arrived);
    unarrived_lots = getSubLot(da.unarrived);

    // distribution
    double tmp;
    iter(arrived_lots, i)
    {
        tmp = (double) arrived_lots[i].qty() / da.upm;
        da.time += tmp;
        arrived_lots[i].addLog("Lot pass DA station");
        result.push_back(arrived_lots[i]);
        if (da.time > 1440) {
            da.finished = true;
            break;
        }
    }

    if (!da.finished) {
        iter(unarrived_lots, i)
        {
            tmp = (double) unarrived_lots[i].qty() / da.upm;

            if (da.time > unarrived_lots[i].queueTime()) {
                unarrived_lots[i].addQueueTime(da.time -
                                               unarrived_lots[i].queueTime());
                unarrived_lots[i].setFcstTime(tmp);
                da.time += tmp;
            } else {
                da.time = unarrived_lots[i].queueTime();
                unarrived_lots[i].setFcstTime(tmp);
                da.time += tmp;
            }

            result.push_back(unarrived_lots[i]);
            if (da.time > 1440) {
                da.finished = true;
                break;
            }
        }
    }

    return result;
}

std::vector<lot_t> da_stations_t::getSubLot(std::vector<lot_t> lots)
{
    std::vector<lot_t> result;
    std::vector<lot_t> temp_lots;
    iter(lots, i)
    {
        if (!lots[i].isSubLot()) {
            temp_lots = lots[i].createSublots();
            result += temp_lots;
        } else {
            result.push_back(lots[i]);
        }
    }
    return result;
}

void da_stations_t::removeAllLots()
{
    for (std::map<std::string, da_station_t>::iterator it =
             _da_stations_container.begin();
         it != _da_stations_container.end(); it++) {
        it->second.arrived.clear();
        it->second.unarrived.clear();
    }
}
