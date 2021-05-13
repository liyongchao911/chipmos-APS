#ifndef __ARRIVAL_H__
#define __ARRIVAL_H__

#include <include/job.h>
#include <string>
#include <vector>

std::vector<lot_t> createLots(std::string wip_file_name,
                              std::string prod_pid_filename,
                              std::string eim,
                              std::string fcst_filename,
                              std::string routelist_filename,
                              std::string queue_time_filename);


void outputReport(std::string filename, std::vector<std::string> report);

#endif
