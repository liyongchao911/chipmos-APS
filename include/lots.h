//
// Created by eugene on 2021/7/5.
//
#ifndef __LOTS_H__
#define __LOTS_H__

#include <vector>
#include <string>
#include <map>

class lots_t{
protected:
    std::vector<lot_t> lots;
    std::map<std::string, std::vector<lot_t*> > wire_lots;
    std::map<std::string, std::vector<lot_t*> > tool_lots;
    std::map<std::string, std::vector<lot_t*> > tool_wire_lots;
    std::map<std::string, int> amount_of_wires;
    std::map<std::string, int> amount_of_tools;
    std::map<std::string, int> getModelsDistribution(std::map<std::string, std::map<std::string, std::vector<entity_t *> > > ents); //location -> amount
    std::map<std::string, int> getModelsDistribution(std::map<std::string, std::vector<entity_t *> > loc_ents);
    bool toolWireLotsHasLots();

    std::vector<lot_t> createLots(std::string wip_file_name, // wip file
                                  std::string prod_pid_bomid_filename, // production -> process_id mapping relation ship
                                  std::string eim_lot_size_filename,
                                  std::string fcst_filename,
                                  std::string routelist_filename,
                                  std::string queue_time_filename,
                                  std::string bomlist_filename,
                                  std::string pid_heatblock_filename,
                                  std::string ems_heatblock_filename,
                                  std::string gw_filename,
                                  std::string bdid_mapping_models_filename,
                                  std::string uph_filename);
    void readWip(std::string filename, std::vector<lot_t> &lots, std::vector<lot_t> &faulty_lots);
    void setPidBomId(std::string filename, std::vector<lot_t> &lots, std::vector<lot_t> &faulty_lots, std::vector<std::string> &wip_report);
    void setLotSize(std::string filename, std::vector<lot_t> &lots, std::vector<lot_t> &faulty_lots, std::vector<std::string> &wip_report);
    void setupRoute(std::string routelist_filename, std::string queuetime_filename, route_t & routes);
    std::vector<lot_t> wb7Filter(std::vector<lot_t> alllots, std::vector<lot_t> &dontcare, route_t routes);
    std::vector<lot_t> queueTimeAndQueue(std::vector<lot_t> lots, std::vector<lot_t> &faulty_lots, std::vector<lot_t> &dontcare, da_stations_t &das, route_t routes, std::vector<std::string> wip_report);
    void setupCanRunModels(std::string bdid_model_mapping_models_filename, std::vector<lot_t> &lots, std::vector<lot_t> &faulty_lots);

    void setPartId(std::string filename, std::vector<lot_t> &lots, std::vector<lot_t> &faulty_lots, std::vector<std::string> &wip_report);
    void setAmountOfWire(std::string filename, std::vector<lot_t> &lots, std::vector<lot_t> &faulty_lots, std::vector<std::string> &wip_report);
    void setPartNo(std::string filename, std::vector<lot_t> &lots, std::vector<lot_t> &faulty_lots, std::vector<std::string> &wip_report);
    void setAmountOfTools(std::string filename, std::vector<lot_t> &lots, std::vector<lot_t> &faulty_lots, std::vector<std::string> &wip_report);
    void setupUph(std::string uph_file_name, std::vector<lot_t> &lots, std::vector<lot_t> & faulty_lots);
public:

    void addLots(std::vector<lot_t> lots);
    std::vector<lot_group_t> round(entities_t machines);
    std::vector<std::vector<lot_group_t> > rounds(entities_t ents);
    inline std::map<std::string, int> amountOfWires() { return amount_of_wires; }
    inline std::map<std::string, int> amountOfTools() { return amount_of_tools; }

    void createLots(std::map<std::string, std::string> files);
};

#endif
