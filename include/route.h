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

#define DA1 

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


    /// route_name map to station map to the 
    std::map<std::string, std::map<std::string, int> > _queue_time;

private:
    /**
     * setupBeforeStation() - set up the number of  stations(nstations) before ...(stations list).
     *
     * the function is used to setup the 7 stations before WB
     */
    void setupBeforeStation(std::string routename,bool remove, int nstations, int opts, ...);


public:
    /**
     * setRoute() - set routes from dataframe
     */
    void setRoute(std::string routename, csv_t dataframe);

    bool isLotInStations(lot_t lot);

};

#endif
