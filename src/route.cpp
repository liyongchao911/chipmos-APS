#include <include/route.h>
#include <vector>
#include <cstdarg>

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

    setupBeforeStation(routename, true, 7, 4, WB1, WB2, WB3, WB4); // WB - 7

}

void route_t::setupBeforeStation(std::string routename, bool remove, int nstations, int nopts, ...){
    std::set<int> opers; 
    std::set<int> station_opers;
    std::set<unsigned int> indexes;

    std::vector<station_t> stations;

    va_list variables;
    va_start(variables, nopts);
    for(int i = 0; i < nopts; ++i){
        opers.insert(va_arg(variables, int));        
    }
    va_end(variables);

     
    int idx;
    for(unsigned int i = 0; i < _routes[routename].size(); ++i){
        if(opers.count(_routes[routename][i].oper) != 0){ // _routes[routename][i].oper is WB or DA
            idx = i - nstations;
            if(idx < 0){
                idx = 0; 
            }
            for(unsigned int j = idx; j <= i; ++j){
                station_opers.insert(_routes[routename][j].oper);
                indexes.insert(j);
                // stations.push_back(_routes[routename][j]);
            }


        }
    }

    _beforeWB[routename] = station_opers;


    if(remove){
        std::vector<int> vec(indexes.begin(), indexes.end());
        iter(vec, i){
            stations.push_back(_routes[routename][i]);
        }
        _routes[routename] = stations;
    }
}

bool route_t::isLotInStations(lot_t lot){
    return _beforeWB[lot.route()].count(lot.oper());
}

