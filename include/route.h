#ifndef __ROUTE_H__
#define __ROUTE_H__

#include <map>
#include <string>
#include <vector>
#include <set>


#include <include/csv.h>
#include <include/job.h>

#define WB1 2200
#define WB2 3200
#define WB3 3400
#define WB4 3600

#define DA1 2070
#define DA2 2130
#define DA3 3130
#define DA4 3330
#define DA5 4130
#define DA6 4330
#define DA7 4530
#define DA8 4730
#define DA9 5130
#define DA10 5330
#define DA11 5530
#define DA12 5730
#define DA13 6130
#define DA14 6330
#define DA15 6530


#define CURE 2080


typedef struct station_t station_t;
struct station_t{
    std::string route_name;
    std::string station_name;
    int oper;
    int seq;
};

class route_t
{
private:
    /// route_name map to its array of stations
    std::map<std::string, std::vector<station_t> > _routes;

    std::map<std::string, std::set<int> > _beforeWB;

    /// route_name map to station map to the queue_time
    std::map<int, std::map<int, int> > _queue_time;

    std::set<int> _wb_stations;
    std::set<int> _da_stations;


private:
    /**
     * setupBeforeStation() - set up the number of  stations(nstations) before ...(stations list).
     *
     * the function is used to setup the 7 stations before WB
     */
    void setupBeforeStation(std::string routename, bool remove, int nstations, int opts, ...);
    
    int findStationIdx(std::string routename, int oper);

public:
    
    route_t();

    /**
     * setRoute() - set routes from dataframe
     */
    void setRoute(std::string routename, csv_t dataframe);

    void setQueueTime(csv_t queue_time_df);

    bool isLotInStations(lot_t lot);
        
    /**
     * return 0 : finished -> lot is in WB
     * return 1 : dispatched -> lot is traverse to DA
     * return 2 : dispatched -> lot is wait on DA
     * return -1 : error
     */
    int calculatQueueTime(lot_t & lot);
};

#endif
