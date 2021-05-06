#ifndef __ROUTE_H__
#define __ROUTE_H__

#include <string>
#include <vector>
#include <map>
#include <include/csv.h>

typedef struct station_t station_t;
struct station_t{
    std::string route_name;
    std::string station_name;
    int oper;
    int seq;
};

class route{
private:
    /// route_name map to its array of stations
    std::map<std::string, std::vector<station_t> > _routes;

    /// route_name map to station map to the 
    std::map<std::string, std::map<std::string, int> > _queue_time;
public:
    /**
     * setRoute() - set routes from dataframe
     */
    void setRoute(std::string routename, csv dataframe);

    /**
     * 
     */
};

#endif
