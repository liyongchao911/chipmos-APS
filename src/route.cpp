#include <include/route.h>

void route_t::setRoute(std::string routename, csv_t dataframe)
{
    std::vector<std::vector<std::string> > data = dataframe.getData();
    std::vector<station_t> stations;
    std::map<std::string, std::string> elements;
    unsigned int size = dataframe.nrows();
    for (unsigned int i = 0; i < size; ++i) {
        elements = dataframe.getElements(i);
        stations.push_back(station_t{.route_name = elements["route_t"],
                                     .station_name = elements["desc"],
                                     .oper = stoi(elements["oper"]),
                                     .seq = stoi(elements["seq"])});
    }

    _routes[routename] = stations;
}
