#ifndef __ARRIVAL_H__
#define __ARRIVAL_H__

#include <include/job.h>
#include <string>
#include <vector>
/**
 * createLots () - create lot by reading a bunch of file
 *
 * ****************************************************************
 *                         prod_pid         pid_bomId
 * WIP -----------> lots -----------> lots ----------->lots -->
 *       |                    |                 |
 *       |                    |                 |
 *       |                    |                 |
 *       v                    v                 v
 *    wip_report         faulty_lot        faulty_lot
 *
 * ****************************************************************
 *
 *      pid_lotSize         wb_7_filter         queueTimeAndQ
 * lots-------------> lots --------------> lots -------------->lots
 *          |                   |                       |
 *          |                   |                       |
 *          v                   v                       v
 *      faulty_lot        dont_care_lots     faulty_lot, dont_care
 *
 * ****************************************************************
 *
 *       setCanRunModels
 * lots ------------------> return
 *           |
 *           |
 *           v
 *      faulty_lot
 *
 * ****************************************************************
 */
std::vector<lot_t> createLots(std::string wip_file_name,
                              std::string prod_pid_filename,
                              std::string eim,
                              std::string fcst_filename,
                              std::string routelist_filename,
                              std::string queue_time_filename,
                              std::string bomlist_filename,
                              std::string heatblock_filename,
                              std::string ems_filename,
                              std::string gw_filename);


void outputReport(std::string filename, std::vector<std::string> report);

#endif
