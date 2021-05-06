#ifndef __ROUTE_H__
#define __ROUTE_H__

#include <include/csv.h>
#include <map>
#include <string>
#include <vector>

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

    /// route_name map to station map to the 
    std::map<std::string, std::map<std::string, int> > _queue_time;
public:
    /**
     * setRoute() - set routes from dataframe
     */
    void setRoute(std::string routename, csv_t dataframe);

    /**
     * 
     */
};

#endif
