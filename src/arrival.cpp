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


int arrivalTime(int argc, const char *argv[])
{
    vector<string> wip_report;
    string err_msg;


    // setup wip
    // Step 1 : read wip.csv
    csv_t wip("wip.csv", "r", true, true);
    wip.trim(" ");
    wip.setHeaders(map<string, string>({{"lot_number", "wlot_lot_number"},
                                        {"qty", "wlot_qty_1"},
                                        {"hold", "wlot_hold"},
                                        {"oper", "wlot_oper"},
                                        {"mvin", "wlot_mvin_perfmd"},
                                        {"recipe", "bd_id"},
                                        {"prod_id", "wlot_prod"}}));

    // Step 2 : read product_find_process_id
    csv_t prod_pid_mapping("product_find_process_id.csv", "r", true, true);
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

    // Step 3 : sublot_size;
    // TODO : read sublot size
    csv_t eim("process_find_lot_size_and_entity.csv", "r", true, true);
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

    // Finally, combine all data
    std::vector<lot_t> alllots;
    std::vector<lot_t> faulty_lots;
    std::vector<lot_t> lots;
    lot_t lot_tmp;
    for (unsigned int i = 0, size = wip.nrows(); i < size; ++i) {
        try {
            // lot_tmp = lot_t(wip.getElements(i));
            lot_tmp = lot_t(wip.getElements(i));
        } catch (std::invalid_argument &e) {
            wip_report.push_back("Lot Entry  " + to_string(i + 2) + " " +
                                 e.what());
            continue;
        }

        try {
            string process_id = prod_pid.at(lot_tmp.prodId());
            lot_tmp.setProcessId(process_id);
        } catch (std::out_of_range &e) {
            wip_report.push_back(
                "Lot Entry " + to_string(i + 2) + ": " + lot_tmp.lotNumber() +
                " has no mapping relationship between its product id(" +
                lot_tmp.prodId() + ") and its process_id");
            continue;
        }

        try {
            string bom_id = prod_bom.at(lot_tmp.prodId());
            lot_tmp.setBomId(bom_id);
        } catch (std::out_of_range &e) {
            wip_report.push_back(
                "Lot Entry" + to_string(i + 2) + ": " + lot_tmp.lotNumber() +
                "has no mapping relation between its product id(" +
                lot_tmp.prodId() + ") and its bom_id");
            continue;
        }
        int lot_size;
        try {
            lot_size = pid_lotsize.at(lot_tmp.processId());
            lot_tmp.setLotSize(lot_size);
        } catch (std::out_of_range &e) {  // catch map::at
            wip_report.push_back(
                "Lot Entry" + to_string(i + 2) + ": " + lot_tmp.lotNumber() +
                "has no mapping relation between its process id(" +
                lot_tmp.processId() + ") and its lot_size");
            continue;
        } catch (std::invalid_argument &e) {
            wip_report.push_back(
                "Lot Entry" + to_string(i + 2) + ": " + lot_tmp.lotNumber() +
                "the lot_size of process id(" + lot_tmp.processId() + ") is" +
                to_string(lot_size) + " which is less than 0");
            continue;
        }

        alllots.push_back(lot_tmp);
    }

    if (alllots.size() == 0) {
        outputReport("wip-report.txt", wip_report);
        exit(-1);
    }

    /*************************************************************************************/

    // setup da_stations_t
    csv_t fcst("fcst.csv", "r", true, true);
    fcst.trim(" ");
    da_stations_t das(fcst, false);


    // setup routes
    route_t routes;
    csv_t routelist_df("routelist.csv", "r", true, true);
    csv_t queue_time("newqueue_time.csv", "r", true, true);
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


    // filter, check if lot is in scheduling plan
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
    

    // route traversal and sum the queue time
    int retval = 0;
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
                    unfinished[i].addLog("Lot is finished traversing the route");
                    finished.push_back(unfinished[i]);
                    break;
                case 2:  // add to DA_arrived
                    unfinished[i].addLog("Lot is waiting on DA station, it is cataloged to arrived");
                    das.addArrivedLotToDA(unfinished[i]);
                    break;
                case 1:  // add to DA_unarrived
                    unfinished[i].addLog("Lot traverse to DA station, it is cataloged to unarrived");
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

    outputReport("wip-report.txt", wip_report);
    return 0;
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
