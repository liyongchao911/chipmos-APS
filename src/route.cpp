#include "include/route.h"
#include <cstdarg>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>
#include "include/infra.h"
#include "include/lot.h"

const int WB_STATIONS[] = {2200, 3200, 3400, 3600, 4200, 4400, 4600, 4800};
const int NUMBER_OF_WB_STATIONS = ARRAY_SIZE(WB_STATIONS, int);

const int DA_STATIONS[] = {2070, 2130, 3130, 3330, 4130, 4300, 4530, 4730,
                           5130, 5330, 5530, 5730, 6130, 6330, 6530};
const int NUMBER_OF_DA_STATIONS = ARRAY_SIZE(DA_STATIONS, int);

// const int CURE_STATIONS[] = {2140, 2150, 2250, 2330, 2405, 2425, 2428, 2472,
// 3140, 3150, 3340, 3350, 3550, 4140, 4150};
const int CURE_STATIONS[] = {2140, 2425, 3140, 3340, 4140};
const int NUMBER_OF_CURE_STATIONS = ARRAY_SIZE(CURE_STATIONS, int);

using namespace std;


route_t::route_t()
{
    vector<int> wb_stations;
    vector<int> da_stations;
    vector<int> cure_stations;

    for (int i = 0, size = ARRAY_SIZE(WB_STATIONS, int); i < size; ++i) {
        wb_stations.push_back(WB_STATIONS[i]);
    }

    for (int i = 0, size = ARRAY_SIZE(DA_STATIONS, int); i < size; ++i) {
        da_stations.push_back(DA_STATIONS[i]);
    }

    for (int i = 0, size = ARRAY_SIZE(CURE_STATIONS, int); i < size; ++i) {
        cure_stations.push_back(CURE_STATIONS[i]);
    }


    _wb_stations = set<int>(wb_stations.begin(), wb_stations.end());
    _da_stations = set<int>(da_stations.begin(), da_stations.end());
    _cure_stations = set<int>(cure_stations.begin(), cure_stations.end());
}

void route_t::setRoute(csv_t all_routes)
{
    // first, get the set of route names
    std::vector<std::string> route_names = all_routes.getColumn("route");
    std::set<std::string> route_list_set(route_names.begin(),
                                         route_names.end());
    route_names =
        std::vector<std::string>(route_list_set.begin(), route_list_set.end());

    // second, call route_t::setRoute for each route respectively
    csv_t df;
    iter(route_names, i)
    {
        df = all_routes.filter("route", route_names[i]);
        setRoute(route_names[i], df);
    }
}

void route_t::setRoute(std::string routename, csv_t dataframe)
{
    std::vector<std::vector<std::string> > data = dataframe.getData();
    std::vector<station_t> stations;
    std::map<std::string, std::string> elements;
    unsigned int size = dataframe.nrows();
    for (unsigned int i = 0; i < size; ++i) {
        elements = dataframe.getElements(i);
        stations.push_back(station_t{.route_name = elements["route"],
                                     .station_name = elements["desc"],
                                     .oper = stoi(elements["oper"]),
                                     .seq = stoi(elements["seq"])});
    }

    _routes[routename] = stations;

    setupBeforeStation(routename, true, 7, 8, WB_STATIONS[0], WB_STATIONS[1],
                       WB_STATIONS[2], WB_STATIONS[3], WB_STATIONS[4],
                       WB_STATIONS[5], WB_STATIONS[6],
                       WB_STATIONS[7]);  // WB - 7
}


void route_t::setQueueTime(csv_t queue_time_df)
{
    unsigned int nrows = queue_time_df.nrows();
    std::map<std::string, std::string> elements;
    std::map<int, double> queue_time;
    int station;
    for (unsigned int i = 0; i < nrows; ++i) {
        elements = queue_time_df.getElements(i);
        station = std::stoi(elements["station"]);
        elements.erase(elements.find("station"));
        for (std::map<std::string, std::string>::iterator it = elements.begin();
             it != elements.end(); it++) {
            queue_time[std::stoi(it->first)] = std::stod(it->second) * 60;
        }
        _queue_time[station] = queue_time;
    }
}

void route_t::setCureTime(csv_t remark_df, csv_t cure_time_df)
{
    cure_time.clear();

    remark_df = remark_df.filter("master_desc", "PTN no");

    map<string, int> cure_time_map;
    for (int i = 0, size = cure_time_df.nrows(); i < size; ++i) {
        map<string, string> elements = cure_time_df.getElements(i);
        cure_time_map[elements["PTN"]] =
            stoi(elements["raising_time"]) + stoi(elements["1st_step_time"]);
    }


    for (int i = 0, size = remark_df.nrows(); i < size; ++i) {
        map<string, string> elements = remark_df.getElements(i);
        string key = elements["process_id"] + "_" + elements["detail_id"];
        if (cure_time_map.count(elements["remark"]) == 0) {
            cure_time[key] = 0;
        } else {
            cure_time[key] = cure_time_map[elements["remark"]];
        }
    }
}

std::vector<station_t> route_t::setupBeforeStation(std::string routename,
                                                   bool remove,
                                                   int nstations,
                                                   int nopts,
                                                   ...)
{
    std::set<int> opers;
    std::set<int> station_opers;
    std::set<unsigned int> indexes;

    std::vector<station_t> stations;

    va_list variables;
    va_start(variables, nopts);
    for (int i = 0; i < nopts; ++i) {
        opers.insert(va_arg(variables, int));
    }
    va_end(variables);


    int idx;
    for (unsigned int i = 0; i < _routes[routename].size(); ++i) {
        if (opers.count(_routes[routename][i].oper) !=
            0) {  // _routes[routename][i].oper is WB or DA
            idx = i - nstations;
            if (idx < 0) {
                idx = 0;
            }
            for (unsigned int j = idx; j <= i; ++j) {
                station_opers.insert(_routes[routename][j].oper);
                indexes.insert(j);
                // stations.push_back(_routes[routename][j]);
            }
        }
    }

    _beforeWB[routename] = station_opers;


    if (remove) {
        std::vector<int> vec(indexes.begin(), indexes.end());
        iter(vec, i) { stations.push_back(_routes[routename][vec[i]]); }
        _routes[routename] = stations;
    }

    return stations;
}

bool route_t::isLotInStations(lot_t lot)
{
    int idx, oper;
    if (_wb_stations.count(lot.oper()) &&
        lot.mvin()) {  // if lot is on WB station  and has moved in
        idx = findStationIdx(lot.route(),
                             lot.oper());  // locate the oper on the route
        if (idx > 0 && (unsigned int) (idx + 1) <
                           _routes[lot.route()]
                               .size()) {  // check if idx is route is resonable
            oper = _routes[lot.route()][++idx].oper;
            return _beforeWB[lot.route()].count(oper);
        } else
            return false;
    }
    return _beforeWB[lot.route()].count(lot.oper());
}

int route_t::findStationIdx(std::string routename, int oper)
{
    int idx = -1;
    iter(_routes[routename], i)
    {
        if (_routes[routename][i].oper == oper) {
            idx = i;
        }
    }

    return idx;
}


/**
 * In this function, the main issue is to determine the begining station of
 * traversal.
 *
 * if lot is in DA and lot isn't moved in, -> dispatch
 * if lot isn't in DA, ->
 *  1. traverse the route to DA and sum the queue time.
 *  2. dispatch
 * if lot is in DA and has been moved in ->
 *  1. begining station is next station.
 *
 */
int route_t::calculateQueueTime(lot_t &lot)
{
    if (lot.isTraversalFinish())
        return 0;

    std::string routename = lot.route();

    // locate the lot in the route
    int idx = findStationIdx(routename, lot.tmp_oper);

    // if can't locate the oper, idx will be less then 0
    if (idx < 0) {
        std::string error = "Can't locate lot.temp_oper(" +
                            std::to_string(lot.tmp_oper) + ") on the route|" +
                            std::to_string(ERROR_INVALID_OPER_IN_ROUTE);
        throw std::logic_error(error);
    }

    // check if lot is in WB
    // the conditions are as follow:
    //  1. if lot.oper == WB
    //  2. if lot.mvin == true
    //
    // if the conditions are true it means that lot is in WB and lot has moved
    // in ! in condition 2,  the lot is considered as lot is finished in WB. To
    // calculate the arrival time, lot is only need to plus the out plan time
    // and the queue time in next stations.
    //
    // in this condition, idx need to plus 1, the lot start from next station
    if (_wb_stations.count(lot.tmp_oper) == 1) {
        if (lot.tmp_mvin) {        // if lot is in WB an has moved in,
            idx += 1;              // move to next station
            lot.tmp_mvin = false;  // mvin = false
        } else {                   // lot is waiting at WB station
            lot.setTraverseFinished();
            return 0;
        }
    }

    // check if lot is in DA
    // if lot is in DA,
    if (_da_stations.count(lot.tmp_oper) == 1) {  // lot is in DA
        if (lot.tmp_mvin &&
            lot.isSubLot()) {  // lot is in DA and mvin and which is sublot
            idx += 1;          // advance
            lot.tmp_mvin = false;
        } else {  // lot is in D/A and hasn't moved in or which still is sublot
            lot.tmp_oper = _routes[routename][++idx].oper;  // advance
            return 2;  // advance and dispatch
        }
    }

    lot.tmp_mvin = false;


    // traverse the route from the begining station idx to D/A or W/B or to the
    // end of route if traverse to the end of route, it means that the lot will
    // not being scheduled to process in W/B station -> it is totally wrong!
    // please check route list or the route.h define macro. maybe _wb_stations
    // doesn't contain the extra W/B stations
    int oper;
    int prev = 0;
    int qt = 0;
    iter_range(_routes[routename], i, idx, _routes[routename].size())
    {
        oper = _routes[routename][i].oper;
        if (_queue_time.count(oper)) {  // check if oper is a big station?
            if (prev) {                 // prev is not null
                if (_queue_time[prev][oper] > 0) {
                    qt += _queue_time[prev][oper];
                    prev = oper;
                } else {
                    std::string error_text =
                        std::to_string(prev) + " -> " + std::to_string(oper) +
                        " is invalid queue time combination, please check "
                        "queue_time's input file. |" +
                        std::to_string(ERROR_INVALID_QUEUE_TIME_COMBINATION);

                    throw std::logic_error(error_text);
                }
            } else {
                prev = oper;
            }

            if (_cure_stations.count(oper) != 0) {
                string process_id = lot.processId();
                int cure_time = getCureTime(process_id, oper);
                string log = "Pass the station(" + to_string(oper) +
                             "), add cure time : " + to_string(cure_time);
                lot.addLog(log, SUCCESS);
                qt += cure_time;
            } else if (_da_stations.count(
                           oper)) {  // oper is a D/A station,  dispatch
                lot.tmp_mvin = false;
                lot.tmp_oper = _routes[routename][i + 1]  // ad
                                   .oper;  // lot traverse to DA station
                lot.addQueueTime(qt);
                return 1;
            } else if (_wb_stations.count(oper)) {  // traverse to W/B station
                lot.tmp_mvin = false;
                lot.tmp_oper = oper;
                lot.addQueueTime(qt);
                lot.setTraverseFinished();
                return 0;
            }
        }
    }

    return -1;
}
