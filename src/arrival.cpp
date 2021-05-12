#include <include/arrival.h>
#include <ctime>
#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include <include/common.h>
#include <include/csv.h>
#include <include/da.h>
#include <include/job.h>
#include <include/route.h>


using namespace std;

void readWip(string filename, vector<lot_t> &lots, vector<string> &wip_report)
{
    // setup wip
    // Step 1 : read wip.csv
    csv_t wip(filename, "r", true, true);
    wip.trim(" ");
    wip.setHeaders(map<string, string>({{"lot_number", "wlot_lot_number"},
                                        {"qty", "wlot_qty_1"},
                                        {"hold", "wlot_hold"},
                                        {"oper", "wlot_oper"},
                                        {"mvin", "wlot_mvin_perfmd"},
                                        {"recipe", "bd_id"},
                                        {"prod_id", "wlot_prod"}}));
    lot_t lot_tmp;
    for (unsigned int i = 0, size = wip.nrows(); i < size; ++i) {
        try {
            lot_tmp = lot_t(wip.getElements(i));
        } catch (std::invalid_argument &e) {
            wip_report.push_back("Lot Entry  " + to_string(i + 2) + " " +
                                 e.what());
            continue;
        }
        lots.push_back(lot_tmp);
    }
}

void setPidBomId(string filename,
                 vector<lot_t> &lots,
                 vector<lot_t> &faulty_lots,
                 vector<string> &wip_report)
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
            wip_report.push_back(err_msg);
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
            wip_report.push_back(err_msg);
            continue;
        }
        result.push_back(lots[i]);
    }

    lots = result;
}

void setLotSize(string filename,
                vector<lot_t> &lots,
                vector<lot_t> &faulty_lots,
                vector<string> &wip_report)
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
            wip_report.push_back(err_msg);
            faulty_lots.push_back(lots[i]);
            continue;
        } catch (std::invalid_argument &e) {
            err_msg = "Lot Entry" + to_string(i + 2) + ": " +
                      lots[i].lotNumber() + "the lot_size of process id(" +
                      lots[i].processId() + ") is" + to_string(lot_size) +
                      " which is less than 0";
            wip_report.push_back(err_msg);
            lots[i].addLog(err_msg);
            faulty_lots.push_back(lots[i]);
            continue;
        }
        result.push_back(lots[i]);
    }
    lots = result;
}

void setupRoute(string routelist, string queuetime, route_t &routes)
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

vector<lot_t> wb_7_filter(vector<lot_t> alllots,
                          vector<lot_t> &faulty_lots,
                          route_t routes)
{
    vector<lot_t> lots;
    iter(alllots, i)
    {
        if (alllots[i].hold()) {
            alllots[i].addLog("Lot is hold");
            faulty_lots.push_back(alllots[i]);
        } else if (!routes.isLotInStations(alllots[i])) {
            alllots[i].addLog("Lot is not in WB - 7");
            faulty_lots.push_back(alllots[i]);
        } else if (alllots[i].qty() <= 0) {
            alllots[i].addLog("Lot's qty <= 0");
            faulty_lots.push_back(alllots[i]);
        } else {
            lots.push_back(alllots[i]);
        }
    }
    return lots;
}

vector<lot_t> queueTimeAndQueue(vector<lot_t> lots,
                                vector<lot_t> &faulty_lots,
                                da_stations_t &das,
                                route_t routes,
                                vector<string> wip_report)
{
    int retval = 0;
    string err_msg;
    std::vector<lot_t> unfinished = lots;
    std::vector<lot_t> finished;
    std::vector<lot_t> dontcare;

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
                    wip_report.push_back(
                        err_msg + "lot information : " + unfinished[i].info());
                    break;
                case 0:  // lot is finished
                    unfinished[i].addLog(
                        "Lot is finished traversing the route");
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
                        "Lot traverse to DA station, it is cataloged to "
                        "unarrived");
                    das.addUnarrivedLotToDA(unfinished[i]);
                    break;
                }
            } catch (std::out_of_range
                         &e) {  // for da_stations_t function member,
                                // addArrivedLotToDA and addUnarrivedLotToDA
                unfinished[i].addLog(e.what());
                faulty_lots.push_back(unfinished[i]);
                wip_report.push_back(e.what() +
                                     std::string("lot information : ") +
                                     unfinished[i].info());
            } catch (std::logic_error &e) {  // for calculateQueueTime
                unfinished[i].addLog(e.what());
                faulty_lots.push_back(unfinished[i]);
                wip_report.push_back(e.what() +
                                     std::string("lot information : ") +
                                     unfinished[i].info());
            }
        }
        unfinished = das.distributeProductionCapacity();
        das.removeAllLots();
    }

    return finished;
}


vector<lot_t> createLots(string wip_file_name,
                         string prod_pid_filename,
                         string eim,
                         string fcst_filename,
                         string routelist_filename,
                         string queue_time_filename)
{
    vector<string> wip_report;
    string err_msg;

    std::vector<lot_t> alllots;
    std::vector<lot_t> faulty_lots;
    std::vector<lot_t> lots;
    readWip(wip_file_name, alllots, wip_report);
    setPidBomId(prod_pid_filename, alllots, faulty_lots, wip_report);
    setLotSize(eim, alllots, faulty_lots, wip_report);

    if (alllots.size() == 0) {
        outputReport("wip-report.txt", wip_report);
        exit(-1);
    }

    /*************************************************************************************/

    // setup da_stations_t
    csv_t fcst(fcst_filename, "r", true, true);
    fcst.trim(" ");
    da_stations_t das(fcst, false);

    route_t routes;
    setupRoute(routelist_filename, queue_time_filename, routes);


    // filter, check if lot is in scheduling plan
    lots = wb_7_filter(alllots, faulty_lots, routes);


    // route traversal and sum the queue time
    vector<lot_t> dontcare;
    lots = queueTimeAndQueue(lots, dontcare, das, routes, wip_report);

    // TODO: output faulty lot
    // TODO: output dontcare lot

    outputReport("wip-report.txt", wip_report);
    return lots;
}

void outputReport(string filename, vector<string> report)
{
    FILE *file = fopen(filename.c_str(), "w");

    if (file) {
        for (unsigned int i = 0; i < report.size(); ++i) {
            fprintf(file, "%s\n", report[i].c_str());
        }
    }

    fclose(file);
}
