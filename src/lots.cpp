#include <assert.h>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <ios>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "include/condition_card.h"
#include "include/csv.h"
#include "include/da.h"
#include "include/infra.h"
#include "include/lot.h"
#include "include/lots.h"
#include "include/route.h"

using namespace std;


#define separateToolWireName(tool_wire_name, t_name, w_name)     \
    t_name = tool_wire_name.substr(0, tool_wire_name.find("_")); \
    w_name = tool_wire_name.substr(tool_wire_name.find("_") + 1);

void lots_t::addLots(std::vector<lot_t> lots)
{
    this->lots = lots;
    std::string part_id, part_no;
    iter(this->lots, i)
    {
        part_id = this->lots[i].part_id();
        part_no = this->lots[i].part_no();
        this->tool_lots[part_no].push_back(&(this->lots[i]));
        this->wire_lots[part_id].push_back(&(this->lots[i]));
        this->tool_wire_lots[part_no + "_" + part_id].push_back(
            &(this->lots[i]));

        amount_of_tools[this->lots[i].part_no()] =
            this->lots[i].getAmountOfTools();
        amount_of_wires[this->lots[i].part_id()] =
            this->lots[i].getAmountOfWires();
    }
}

std::map<std::string, int> lots_t::initializeModelDistribution(
    std::map<std::string, std::vector<entity_t *> > loc_ents)
{
    std::map<std::string, int> _;
    for (auto &loc_ent : loc_ents) {
        _[loc_ent.first] = 0;
    }
    return _;
}

std::vector<lot_group_t> lots_t::selectGroups(int max)
{
    // max : 50 sets;
    std::vector<lot_group_t> groups;
    std::vector<lot_group_t> selected_groups;
    for (auto &tool_wire_lot : tool_wire_lots) {
        groups.push_back(
            lot_group_t{.wire_tools_name = tool_wire_lot.first,
                        .lot_amount = tool_wire_lot.second.size()});
    }
    std::sort(groups.begin(), groups.end(), lotGroupCmp);
    int selected_group_number = (groups.size() > max ? max : groups.size());
    for (int i = 0; i < selected_group_number; ++i) {
        if (groups[i].lot_amount > 0) {
            selected_groups.push_back(groups[i]);
        }
    }
    return selected_groups;
}

void lots_t::setupToolWireAmount(vector<lot_group_t> &selected_groups)
{
    // count the number of lots wanting to use its tool and wire
    std::map<std::string, int> sta_tools;
    std::map<std::string, int> sta_wires;
    std::string t, w, t_w;
    iter(selected_groups, i)
    {
        t_w = selected_groups[i].wire_tools_name;
        separateToolWireName(t_w, t, w);
        selected_groups[i].wire_name = w;
        selected_groups[i].tool_name = t;
        if (sta_tools.count(t) == 0) {
            sta_tools[t] = 0;
        }
        if (sta_wires.count(w) == 0) {
            sta_wires[w] = 0;
        }

        sta_tools[t] += selected_groups[i].lot_amount;
        sta_wires[w] += selected_groups[i].lot_amount;
    }

    double ratio;
    iter(selected_groups, i)
    {
        ratio = selected_groups[i].lot_amount /
                (double) sta_tools.at(selected_groups[i].tool_name);
        selected_groups[i].tool_amount =
            ratio * amount_of_tools.at(selected_groups[i].tool_name);

        ratio = selected_groups[i].lot_amount /
                (double) sta_wires.at(selected_groups[i].wire_name);
        selected_groups[i].wire_amount =
            ratio * amount_of_wires.at(selected_groups[i].wire_name);

        selected_groups[i].machine_amount =
            selected_groups[i].tool_amount > selected_groups[i].wire_amount
                ? selected_groups[i].wire_amount
                : selected_groups[i].tool_amount;
    }
}

map<string, int> lots_t::bdidStatistic(vector<lot_t *> lots)
{
    map<string, int> ret;
    iter(lots, i)
    {
        if (ret.count(lots[i]->recipe()) == 0) {
            ret[lots[i]->recipe()] = 1;
        } else
            ret[lots[i]->recipe()] += 1;
    }
    return ret;
}

map<string, int> lots_t::modelStatistic(
    vector<lot_t *> lots,
    map<string, vector<entity_t *> > loc_ents)
{
    map<string, int> ret;
    ret = initializeModelDistribution(loc_ents);
    iter(lots, i)
    {
        std::vector<std::string> can_run_locations =
            lots[i]->getCanRunLocations();
        iter(can_run_locations, j) { ret[can_run_locations[j]] += 1; }
    }
    return ret;
}

vector<lot_group_t> lots_t::round(entities_t machines)
{
    // loc_ents is a map container which maps location to several entities
    map<string, vector<entity_t *> > loc_ents = machines.getLocEntity();

    // model_location is a map conainer which store the model and its location
    map<string, vector<string> > model_location = machines.getModelLocation();

    vector<lot_group_t> selected_groups;

    lot_group_t test_lot_group;

    // initialize
    iter(lots, i)
    {
        lots[i].clearCanRunLocation();
        lots[i].setCanRunLocation(model_location);
    }
    machines.reset();
    selected_groups = selectGroups(20);

    // setup the number of tool and the number of wire
    setupToolWireAmount(selected_groups);

    // models statistic
    string tool_wire_name;
    iter(selected_groups, k)
    {
        if (selected_groups[k].machine_amount > 0) {
            std::vector<lot_t *> lots =
                this->tool_wire_lots[selected_groups[k].wire_tools_name];

            selected_groups[k].models_statistic =
                modelStatistic(lots, loc_ents);
            selected_groups[k].bdid_statistic = bdidStatistic(lots);
        }
    }


    iter(selected_groups, i)
    {
        // int lot_amount = selected_groups[i].lot_amount;
        // int machine_number = lot_amount > 4 ? lot_amount >> 2 : 1;
        // machine_number = machine_number >  selected_groups[i].machine_amount ? selected_groups[i].machine_amount : machine_number;
        // selected_groups[i].entities = machines.randomlyGetEntitiesByLocations(selected_groups[i].models_statistic, selected_groups[i].bdid_statistic, machine_number);
        // if (selected_groups[i].lot_amount < 10)
        //     machine_number = (3 > selected_groups[i].machine_amount
        //                           ? selected_groups[i].machine_amount
        //                           : 3);
        // else if (selected_groups[i].machine_amount > 20) {
        //     machine_number = selected_groups[i].lot_amount / 10;
        //     machine_number = (selected_groups[i].lot_amount / 10 > 20
        //                           ? 20
        //                           : selected_groups[i].machine_amount);
        // } else {
        //     machine_number = selected_groups[i].machine_amount;
        //     selected_groups[i].entities =
        //         machines.randomlyGetEntitiesByLocations(
        //             selected_groups[i].models_statistic,
        //             selected_groups[i].bdid_statistic, machine_number);
        // }
        
        if (selected_groups[i].lot_amount < 10)
            selected_groups[i].entities =
                machines.randomlyGetEntitiesByLocations(
                    selected_groups[i].models_statistic,
                    selected_groups[i].bdid_statistic,
                    selected_groups[i].machine_amount > 10
                        ? 3
                        : selected_groups[i].machine_amount);
        else
            selected_groups[i].entities =
                machines.randomlyGetEntitiesByLocations(
                    selected_groups[i].models_statistic,
                    selected_groups[i].bdid_statistic,
                    selected_groups[i].machine_amount);
    }

    test_lot_group = selected_groups[0];


    std::vector<lot_t *> failed;
    std::vector<lot_t *> successful;
    std::set<entity_t *> entities_set;
    std::vector<lot_group_t> ret;
    iter(selected_groups, i)
    {
        // check if the entity is selected by at least 2 groups
        iter(selected_groups[i].entities, j)
        {
            if (entities_set.count(selected_groups[i].entities[j]) == 0) {
                entities_set.insert(selected_groups[i].entities[j]);
            } else {
                string err = "group " + to_string(i) + " is duplicated!\n";
                throw logic_error(err);
            }
        }

        // update the machine_amount.
        selected_groups[i].machine_amount = selected_groups[i].entities.size();
        tool_wire_name = selected_groups[i].wire_tools_name;
        std::vector<lot_t *> lots = tool_wire_lots[tool_wire_name];

        successful.clear();
        failed.clear();
        iter(lots, j)
        {
            bool found = false;  // check there is a acceptable entity
            iter(selected_groups[i].entities, k)
            {
                if (lots[j]->addCanRunEntity(selected_groups[i].entities[k])) {
                    found = true;
                }
            }
            if (found)
                successful.push_back(lots[j]);
            else {
                failed.push_back(lots[j]);
            }
        }
        tool_wire_lots[tool_wire_name] = failed;
        selected_groups[i].lots = successful;
        if (selected_groups[i].lots.size() > 0) {
            ret.push_back(selected_groups[i]);
        }
    }

    return ret;
}

bool lots_t::toolWireLotsHasLots()
{
    for (std::map<std::string, std::vector<lot_t *> >::iterator it =
             tool_wire_lots.begin();
         it != tool_wire_lots.end(); it++) {
        if (it->second.size())
            return true;
    }
    return false;
}

std::vector<std::vector<lot_group_t> > lots_t::rounds(entities_t ents)
{
    std::vector<std::vector<lot_group_t> > round_groups;
    while (toolWireLotsHasLots())
        round_groups.push_back(round(ents));
    return round_groups;
}


void lots_t::readWip(string filename,
                     vector<lot_t> &lots,
                     vector<lot_t> &faulty_lots)
{
    // setup wip
    // Step 1 : read wip.csv
    csv_t wip(filename, "r", true, true);
    wip.dropNullRow();
    wip.trim(" ");
    wip.setHeaders(map<string, string>({{"lot_number", "wlot_lot_number"},
                                        {"qty", "wlot_qty_1"},
                                        {"hold", "wlot_hold"},
                                        {"oper", "wlot_oper"},
                                        {"mvin", "wlot_mvin_perfmd"},
                                        {"recipe", "bd_id"},
                                        {"prod_id", "wlot_prod"},
                                        {"urgent_code", "urgent_code"},
                                        {"customer", "wlot_crt_dat_al_1"}}));
    lot_t lot_tmp;
    for (unsigned int i = 0, size = wip.nrows(); i < size; ++i) {
        lot_tmp = lot_t(wip.getElements(i));
        if (lot_tmp.checkFormation()) {
            lots.push_back(lot_tmp);
        } else {
            faulty_lots.push_back(lot_tmp);
        }
    }
}

void lots_t::setPidBomId(string filename,
                         vector<lot_t> &lots,
                         vector<lot_t> &faulty_lots)
{
    // Step 2 : read product_find_process_id
    // csv_t prod_pid_mapping("product_find_process_id.csv", "r", true, true);
    csv_t prod_pid_mapping(filename, "r", true, true);
    prod_pid_mapping.trim(" ");
    prod_pid_mapping.setHeaders(
        map<string, string>({{"prod_id", "product"},
                             {"process_id", "process_id"},
                             {"bom_id", "bom_id"}}));
    map<string, string> prod_pid;
    map<string, string> prod_bom;
    for (unsigned int i = 0; i < prod_pid_mapping.nrows(); ++i) {
        map<string, string> tmp = prod_pid_mapping.getElements(i);
        prod_pid[tmp["prod_id"]] = tmp["process_id"];
        prod_bom[tmp["prod_id"]] = tmp["bom_id"];
    }

    vector<lot_t> result;

    string err_msg;

    iter(lots, i)
    {
        try {
            string process_id = prod_pid.at(lots[i].prodId());
            lots[i].setProcessId(process_id);
        } catch (std::out_of_range &e) {
            err_msg = "Lot Entry " + to_string(i + 2) + ": " +
                      lots[i].lotNumber() +
                      " has no mapping relationship between its product id(" +
                      lots[i].prodId() + ") and its process_id";
            lots[i].addLog(err_msg);
            faulty_lots.push_back(lots[i]);
            continue;
        }

        try {
            string bom_id = prod_bom.at(lots[i].prodId());
            lots[i].setBomId(bom_id);
        } catch (std::out_of_range &e) {
            err_msg = "Lot Entry" + to_string(i + 2) + ": " +
                      lots[i].lotNumber() +
                      "has no mapping relation between its product id(" +
                      lots[i].prodId() + ") and its bom_id";
            lots[i].addLog(err_msg);
            faulty_lots.push_back(lots[i]);
            continue;
        }
        result.push_back(lots[i]);
    }

    lots = result;
}

void lots_t::setLotSize(string filename,
                        vector<lot_t> &lots,
                        vector<lot_t> &faulty_lots)
{
    // Step 3 : sublot_size;
    // TODO : read sublot size
    // csv_t eim("process_find_lot_size_and_entity.csv", "r", true, true);
    csv_t eim(filename, "r", true, true);
    eim.trim(" ");
    eim.setHeaders(map<string, string>({{"process_id", "process_id"},
                                        {"desc", "master_desc"},
                                        {"lot_size", "remark"}}));
    csv_t pid_lotsize_df;
    pid_lotsize_df = eim.filter("desc", "Lot Size");
    map<string, int> pid_lotsize;
    for (unsigned int i = 0; i < pid_lotsize_df.nrows(); ++i) {
        map<string, string> tmp = pid_lotsize_df.getElements(i);
        pid_lotsize[tmp["process_id"]] = stoi(tmp["lot_size"]);
    }

    vector<lot_t> result;
    int lot_size;
    string err_msg;
    iter(lots, i)
    {
        try {
            lot_size = pid_lotsize.at(lots[i].processId());
            lots[i].setLotSize(lot_size);
        } catch (std::out_of_range &e) {  // catch map::at
            err_msg =
                ("Lot Entry" + to_string(i + 2) + ": " + lots[i].lotNumber() +
                 "has no mapping relation between its process id(" +
                 lots[i].processId() + ") and its lot_size");
            lots[i].addLog(err_msg);
            faulty_lots.push_back(lots[i]);
            continue;
        } catch (std::invalid_argument &e) {
            err_msg = "Lot Entry" + to_string(i + 2) + ": " +
                      lots[i].lotNumber() + "the lot_size of process id(" +
                      lots[i].processId() + ") is" + to_string(lot_size) +
                      " which is less than 0";
            lots[i].addLog(err_msg);
            faulty_lots.push_back(lots[i]);
            continue;
        }
        result.push_back(lots[i]);
    }
    lots = result;
}

void lots_t::setupRoute(string routelist, string queuetime, route_t &routes)
{
    // setup routes
    csv_t routelist_df(routelist, "r", true, true);
    csv_t queue_time(queuetime, "r", true, true);
    routelist_df.trim(" ");
    routelist_df.setHeaders(
        map<string, string>({{"route", "wrto_route"},
                             {"oper", "wrto_oper"},
                             {"seq", "wrto_seq_num"},
                             {"desc", "wrto_opr_shrt_desc"}}));
    routes.setQueueTime(queue_time);


    vector<vector<string> > data = routelist_df.getData();
    vector<string> routenames = routelist_df.getColumn("route");
    set<string> routelist_set(routenames.begin(), routenames.end());
    routenames = vector<string>(routelist_set.begin(), routelist_set.end());

    // setRoute -> get wb - 7
    csv_t df;
    iter(routenames, i)
    {
        df = routelist_df.filter("route", routenames[i]);
        routes.setRoute(routenames[i], df);
    }
}

vector<lot_t> lots_t::wb7Filter(vector<lot_t> alllots,
                                vector<lot_t> &dontcare,
                                route_t routes)
{
    vector<lot_t> lots;
    iter(alllots, i)
    {
        if (alllots[i].hold()) {
            alllots[i].addLog("Lot is hold");
            dontcare.push_back(alllots[i]);
        } else if (!routes.isLotInStations(alllots[i])) {
            alllots[i].addLog("Lot is not in WB - 7");
            dontcare.push_back(alllots[i]);
        } else {
            lots.push_back(alllots[i]);
        }
    }
    return lots;
}

vector<lot_t> lots_t::queueTimeAndQueue(vector<lot_t> lots,
                                        vector<lot_t> &faulty_lots,
                                        vector<lot_t> &dontcare,
                                        da_stations_t &das,
                                        route_t routes)
{
    int retval = 0;
    string err_msg;
    std::vector<lot_t> unfinished = lots;
    std::vector<lot_t> finished;

    // std::string trace_lot_number("P16ABB5");

    while (unfinished.size()) {
        iter(unfinished, i)
        {
            try {
                retval = routes.calculateQueueTime(unfinished[i]);
                switch (retval) {
                case -1:  // error
                    err_msg =
                        "Error occures on routes.calculateQueueTime, the "
                        "reason is the lot can't reach W/B satation.";
                    unfinished[i].addLog(err_msg);
                    faulty_lots.push_back(unfinished[i]);
                    break;
                case 0:  // lot is finished
                    unfinished[i].addLog("Lot finishes traversing the route");
                    finished.push_back(unfinished[i]);
                    break;
                case 2:  // add to DA_arrived
                    unfinished[i].addLog(
                        "Lot is waiting on DA station, it is cataloged to "
                        "arrived");
                    das.addArrivedLotToDA(unfinished[i]);
                    break;
                case 1:  // add to DA_unarrived
                    unfinished[i].addLog(
                        "Lot traverses to DA station, it is cataloged to "
                        "unarrived");
                    das.addUnarrivedLotToDA(unfinished[i]);
                    break;
                }
            } catch (std::out_of_range
                         &e) {  // for da_stations_t function member,
                                // addArrivedLotToDA and addUnarrivedLotToDA
                unfinished[i].addLog(e.what());
                faulty_lots.push_back(unfinished[i]);
            } catch (std::logic_error &e) {  // for calculateQueueTime
                unfinished[i].addLog(e.what());
                faulty_lots.push_back(unfinished[i]);
            }
        }
        unfinished = das.distributeProductionCapacity();
        das.removeAllLots();
    }

    dontcare += das.getParentLots();

    return finished;
}

void lots_t::setCanRunModels(string bdid_model_mapping_models_filename,
                             vector<lot_t> &lots,
                             vector<lot_t> &faulty_lots)
{
    condition_cards_h cards(12, "UTC1000", "UTC1000S", "UTC2000", "UTC2000S",
                            "UTC3000", "UTC5000S", "Maxum Base", "Maxum Plus",
                            "Maxum Ultra", "Iconn", "Iconn Plus", "RAPID");
    cards.addMapping("Maxum (Ultra)", 2, "Maxum", "Maxum-Ultra");
    cards.readBdIdModelsMappingFile(bdid_model_mapping_models_filename);
    vector<lot_t> result;
    iter(lots, i)
    {
        try {
            lots[i].setCanRunModels(
                cards.getModels(lots[i].recipe(), lots[i].tmp_oper).models);
            result.push_back(lots[i]);
        } catch (out_of_range &e) {
            lots[i].addLog("Lot has no matched condition card");
            faulty_lots.push_back(lots[i]);
        }
    }
    lots = result;
}

void lots_t::setPartId(string filename,
                       vector<lot_t> &lots,
                       vector<lot_t> &faulty_lots)
{
    // filename = "BomList"
    csv_t bomlist(filename, "r", true, true);
    bomlist.trim(" ");
    bomlist.setHeaders(map<string, string>(
        {{"bom_id", "bom_id"}, {"oper", "oper"}, {"part_id", "part_id"}}));
    map<int, map<string, string> > bom_oper_part;
    for (unsigned int i = 0; i < bomlist.nrows(); ++i) {
        map<string, string> tmp = bomlist.getElements(i);
        // if "oper" is WB, then get its part_id.
        std::set<int> opers = {WB1, WB2, WB3, WB4};
        int oper_int = stoi(tmp["oper"]);
        if (opers.count(oper_int) != 0) {
            bom_oper_part[oper_int][tmp["bom_id"]] = tmp["part_id"];
            // bom_part[tmp["bom_id"]] = tmp["part_id"];
        }
    }

    vector<lot_t> result;

    string err_msg;

    iter(lots, i)
    {
        try {
            string part_id =
                bom_oper_part.at(lots[i].tmp_oper).at(lots[i].bomId());
            ;
            lots[i].setPartId(part_id);
        } catch (std::out_of_range &e) {
            err_msg = "Lot Entry " + to_string(i + 2) + ": " +
                      lots[i].lotNumber() +
                      " has no mapping relationship between its oper :" +
                      to_string(lots[i].tmp_oper) + " bom id(" +
                      lots[i].bomId() + ") and its part_id";
            lots[i].addLog(err_msg);
            faulty_lots.push_back(lots[i]);
            continue;
        }

        result.push_back(lots[i]);
    }

    lots = result;
}

void lots_t::setAmountOfWire(string filename,
                             vector<lot_t> &lots,
                             vector<lot_t> &faulty_lots)
{
    // filename = "GW Inventory.csv"
    csv_t gw(filename, "r", true, true);
    gw.trim(" ");
    gw.setHeaders(map<string, string>(
        {{"gw_part_no", "gw_part_no"}, {"roll_length", "roll_length"}}));
    map<string, int> part_roll;
    for (unsigned int i = 0; i < gw.nrows(); ++i) {
        map<string, string> tmp = gw.getElements(i);
        if (part_roll.count(tmp["gw_part_no"]) == 0) {
            part_roll[tmp["gw_part_no"]] = 0;
        }
        if (stod(tmp["roll_length"]) >= 1000.0) {
            part_roll[tmp["gw_part_no"]] += 1;
        }
    }

    vector<lot_t> result;

    string err_msg;

    iter(lots, i)
    {
        try {
            int amountOfWires = part_roll.at(lots[i].part_id());
            if (amountOfWires > 0) {
                lots[i].setAmountOfWires(amountOfWires);
                result.push_back(lots[i]);
            } else {
                lots[i].addLog("There is no wire.");
                faulty_lots.push_back(lots[i]);
            }
        } catch (std::out_of_range &e) {
            err_msg = "Lot Entry " + to_string(i + 2) + ": " +
                      lots[i].lotNumber() +
                      " has no mapping relationship between its part id(" +
                      lots[i].part_id() + ") and its roll_length";
            lots[i].addLog(err_msg);
            faulty_lots.push_back(lots[i]);
        }
    }

    lots = result;
}

void lots_t::setPartNo(string filename,
                       vector<lot_t> &lots,
                       vector<lot_t> &faulty_lots)
{
    // filename = "Process find heatblock.csv"
    csv_t heatblock(filename, "r", true, true);
    heatblock.trim(" ");
    heatblock.setHeaders(map<string, string>(
        {{"process_id", "process_id"}, {"remark", "remark"}}));
    map<string, string> pid_remark;
    for (unsigned int i = 0; i < heatblock.nrows(); ++i) {
        map<string, string> tmp = heatblock.getElements(i);
        // get substring of remark from first to "("
        string str = tmp["remark"];
        if (str[0] == 'A') {  // if remark == "Compression Mold" or "O/S
                              // Xray檢驗___/20ea", then it shouldn't be used.
            str = str.substr(0, str.find(" "));
            if (str.find("(") != std::string::npos) {
                str = str.substr(0, str.find("("));
            }
            pid_remark[tmp["process_id"]] = str;
        }
    }

    vector<lot_t> result;

    string err_msg;

    iter(lots, i)
    {
        try {
            string part_no = pid_remark.at(lots[i].processId());
            lots[i].setPartNo(part_no);
        } catch (std::out_of_range &e) {
            err_msg = "Lot Entry " + to_string(i + 2) + ": " +
                      lots[i].lotNumber() +
                      " has no mapping relationship between its process id(" +
                      lots[i].processId() + ") and its remark";
            lots[i].addLog(err_msg);
            faulty_lots.push_back(lots[i]);
            continue;
        }

        result.push_back(lots[i]);
    }

    lots = result;
}

void lots_t::setAmountOfTools(string filename,
                              vector<lot_t> &lots,
                              vector<lot_t> &faulty_lots)
{
    // filename = "EMS Heatblock data.csv"
    csv_t ems(filename, "r", true, true);
    ems.trim(" ");
    ems.setHeaders(map<string, string>(
        {{"part_no", "part_no"}, {"qty1", "qty1"}, {"qty3", "qty3"}}));
    map<string, int> pno_qty;
    for (unsigned int i = 0; i < ems.nrows(); ++i) {
        map<string, string> tmp = ems.getElements(i);
        int qty1_int = stoi(tmp["qty1"]);
        int qty3_int = stoi(tmp["qty3"]);
        if (qty1_int <= qty3_int) {
            pno_qty[tmp["part_no"]] = qty1_int;
        } else {
            pno_qty[tmp["part_no"]] = qty3_int;
        }
    }

    vector<lot_t> result;

    string err_msg;

    int amountOfTool;
    iter(lots, i)
    {
        try {
            amountOfTool = pno_qty.at(lots[i].part_no());

            if (amountOfTool > 0) {
                lots[i].setAmountOfTools(amountOfTool);
                result.push_back(lots[i]);
            } else {
                lots[i].addLog("There is no tool for this lot.");
                faulty_lots.push_back(lots[i]);
            }
        } catch (std::out_of_range &e) {
            // if part_no has no mapping qty1 and qty3, its number of tool
            // should be 0, and should be recorded which one it is.

            err_msg = "Lot Entry " + to_string(i + 2) + ": " +
                      lots[i].lotNumber() +
                      " has no mapping relationship between its part_no(" +
                      lots[i].part_no() + ") and its qty1 and qty3";
            lots[i].addLog(err_msg);
            faulty_lots.push_back(lots[i]);
        }
    }

    lots = result;
    return;
}

void lots_t::setUph(string uph_file_name,
                    vector<lot_t> &lots,
                    vector<lot_t> &faulty_lots)
{
    csv_t uph_csv(uph_file_name, "r", true, true);
    uph_csv.trim(" ");
    uph_csv.setHeaders(map<string, string>({{"oper", "OPER"},
                                            {"cust", "CUST"},
                                            {"recipe", "B/D#"},
                                            {"model", "MODEL"},
                                            {"uph", "G.UPH"}}));
    bool retval = 0;
    vector<lot_t> temp;
    vector<lot_t> maybe_faulty;
    vector<lot_t> result;
    iter(lots, i)
    {
        retval = lots[i].setUph(uph_csv);
        if (retval) {
            result.push_back(lots[i]);
        } else {
            faulty_lots.push_back(lots[i]);
        }
    }

    lots = result;
}

void lots_t::createLots(map<string, string> files)
{
    this->lots = createLots(
        files["wip"], files["pid_bomid"], files["lot_size"], files["fcst"],
        files["routelist"], files["queue_time"], files["bom_list"],
        files["pid_heatblock"], files["ems_heatblock"], files["gw_inventory"],
        files["bdid_model_mapping"], files["uph"]);
    addLots(lots);
}

vector<lot_t> lots_t::createLots(
    string wip_file_name,            // wip
    string prod_pid_bomid_filename,  // pid_bomid
    string eim_lot_size_filename,    // lot_size
    string fcst_filename,            // fcst
    string routelist_filename,       // route list
    string queue_time_filename,      // queue time
    string bomlist_filename,         // bom list
    string pid_heatblock_filename,   // pid_heatblock mapping file
    string ems_heatblock_filename,   // ems heatblock for the number of tools
    string gw_filename,              // gw_inventory for the number of wires
    string bdid_mapping_models_filename,
    string uph_filename)
{
    string err_msg;

    std::vector<lot_t> alllots;
    std::vector<lot_t> faulty_lots;
    std::vector<lot_t> lots;
    vector<lot_t> dontcare;

    // setup da_stations_t
    csv_t fcst(fcst_filename, "r", true, true);
    fcst.trim(" ");
    da_stations_t das(fcst);

    route_t routes;
    setupRoute(routelist_filename, queue_time_filename, routes);

    // start creating lots
    readWip(wip_file_name, alllots, faulty_lots);

    setPidBomId(prod_pid_bomid_filename, alllots, faulty_lots);
    setLotSize(eim_lot_size_filename, alllots, faulty_lots);

    // TODO : setPartId, setPartNo, setAmountOfTools, setAmountOfWires
    lots = wb7Filter(
        alllots, dontcare,
        routes);  // check each lot if it is in scheduling plan (WB - 7)

    // route traversal
    // sum the queue time and distribute the production capacity
    lots = queueTimeAndQueue(lots, faulty_lots, dontcare, das, routes);
    /*************************************************************************************/

    // setPartId
    setPartId(bomlist_filename, lots, faulty_lots);
    setAmountOfWire(gw_filename, lots, faulty_lots);
    // setPartNo
    setPartNo(pid_heatblock_filename, lots, faulty_lots);
    setAmountOfTools(ems_heatblock_filename, lots, faulty_lots);

    setCanRunModels(bdid_mapping_models_filename, lots, faulty_lots);
    setUph(uph_filename, lots, faulty_lots);


    // output faulty lots
    csv_t faulty_lots_csv("faulty_lots.csv", "w");
    iter(faulty_lots, i) { faulty_lots_csv.addData(faulty_lots[i].data()); }
    faulty_lots_csv.write();
    // output dontcare lots
    csv_t dontcare_lots_csv("dontcare.csv", "w");
    iter(dontcare, i) { dontcare_lots_csv.addData(dontcare[i].data()); }
    dontcare_lots_csv.write();

    // output lots
    csv_t lots_csv("lots.csv", "w");
    iter(lots, i) { lots_csv.addData(lots[i].data()); }
    lots_csv.write();

    // ouput wip
    csv_t wip_csv("out.csv", "w");
    iter(faulty_lots, i) { wip_csv.addData(faulty_lots[i].data()); }
    iter(dontcare, i) { wip_csv.addData(dontcare[i].data()); }
    iter(lots, i) { wip_csv.addData(lots[i].data()); }
    wip_csv.write();

    return lots;
}
