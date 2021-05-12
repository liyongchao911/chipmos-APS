#ifndef __DA_H__
#define __DA_H__

#include <include/common.h>
#include <include/csv.h>
#include <include/job.h>
#include <vector>

typedef struct da_station_t {
    double fcst;
    double act;
    double remain;
    double upm;
    double time;
    bool finished;
    std::vector<lot_t> arrived;
    std::vector<lot_t> unarrived;
} da_station_t;


class da_stations_t
{
private:
    std::map<std::string, da_station_t> _da_stations_container;

    std::vector<lot_t> daDistributeCapacity(
        da_station_t &da);  // for single da_station

    std::vector<lot_t> getSubLot(std::vector<lot_t> lots);

public:
    da_stations_t(csv_t fcst, bool strict = false);

    /**
     * setFcst ()
     *
     * return 0 : -n
     */
    int setFcst(csv_t fcst, bool strict = false);

    bool addArrivedLotToDA(lot_t &lot);
    bool addUnarrivedLotToDA(lot_t &lot);


    std::vector<lot_t> distributeProductionCapacity();
};

#endif
