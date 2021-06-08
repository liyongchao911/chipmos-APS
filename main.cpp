#include <ctime>
#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>

#include <include/arrival.h>
#include <include/condition_card.h>
#include <include/csv.h>
#include <include/da.h>
#include <include/entity.h>
#include <include/infra.h>
#include <include/lot.h>
#include <include/route.h>
#include <include/population.h>


using namespace std;

round_t createARound(vector<lot_group_t> group);
unsigned int convertEntityNameToUInt(string name);

int main(int argc, const char *argv[])
{
    lots_t lots;
    lots.addLots(createLots("WipOutPlanTime_.csv", "product_find_process_id.csv",
                   "process_find_lot_size_and_entity.csv", "fcst.csv",
                   "routelist.csv", "newqueue_time.csv",
                   "BOM List_20210521.csv", "Process find heatblock.csv",
                   "EMS Heatblock data.csv", "GW Inventory.csv"));
    
    // csv_t out("out.csv", "w");
    // iter(lots, i) { out.addData(lots[i].data()); }
    // out.write();

    csv_t machine_csv("machines.csv", "r", true, true);
    machine_csv.trim(" ");
    machine_csv.setHeaders(map<string, string>({{"entity", "ENTITY"},
                                                {"model", "MODEL"},
                                                {"recover_time", "OUTPLAN"}}));

    csv_t location_csv("locations.csv", "r", true, true);
    location_csv.trim(" ");
    location_csv.setHeaders(
        map<string, string>({{"entity", "Entity"}, {"location", "Location"}}));


    char *text = strdup("2020/12/19 10:50");
    entities_t machines(text);
    machines.addMachines(machine_csv, location_csv);
    

    vector<vector<lot_group_t> > round_groups = lots.rounds(machines);

    vector<round_t> rounds;
    iter(round_groups, i){
        rounds.push_back(createARound(round_groups[i]));
    }

    return 0;
}


unsigned int convertEntityNameToUInt(string name){
    union{
        char text[4];
        unsigned int number;
    }data;
    string substr = name.substr(name.length() - 4);
    strncpy(data.text, substr.c_str(), 4);
    return data.number;
}

round_t createARound(vector<lot_group_t> group){
    int AMOUNT_OF_JOBS = 0;
    vector<lot_t *> lots;
    iter(group, i){
        lots += group[i].lots;
    }
    AMOUNT_OF_JOBS = lots.size();
    job_t *jobs = (job_t *)malloc(sizeof(job_t) * AMOUNT_OF_JOBS);
    process_time_t ** pts = (process_time_t**)malloc(sizeof(process_time_t*)*AMOUNT_OF_JOBS);
    int *size_of_pt = (int*)malloc(sizeof(int)*AMOUNT_OF_JOBS); 
    
    iter(lots, i){
        jobs[i] = lots[i]->job();
        vector<string> can_run_ents = lots[i]->getCanRunEntities();   
        map<string, double> ent_process_time = lots[i]->getEntitiyProcessTime();
        pts[i] = (process_time_t *)malloc(sizeof(process_time_t)* can_run_ents.size());
        size_of_pt[i] = can_run_ents.size();
        iter(can_run_ents, j){
            pts[i][j].machine_no = convertEntityNameToUInt(can_run_ents[j]);
            pts[i][j].process_time = ent_process_time[can_run_ents[j]];
        }
    }
    round_t round = round_t{
        .AMOUNT_OF_JOBS = AMOUNT_OF_JOBS,
        .jobs = jobs,
        .process_times = pts,
        .size_of_process_times = size_of_pt
    };
    
    return round;
}
