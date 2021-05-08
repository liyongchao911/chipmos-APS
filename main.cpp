#include <iostream>
#include <map>
#include <set>

#include <include/common.h>
#include <include/csv.h>
#include <include/route.h>
#include <include/job.h>


using namespace std;

int main(int argc, const char *argv[]){
    
    csv_t wip("wip.csv", "r", true, true);
    map<string, string> wipheader = {
        {"lot_number", "wlot_lot_number"},
        {"qty", "wlot_qty_1"},
        {"hold", "wlot_hold"},
        {"oper", "wlot_oper"},
        {"mvin", "wlot_mvin_perfmd"}
    };
    wip.setHeaders(wipheader);
    wip.trim(" ");
    

    std::vector<lot_t> alllots;
    std::vector<lot_t> lots;
    for(unsigned int i = 0, size = wip.nrows(); i < size; ++i){
        alllots.push_back(lot_t(wip.getElements(i)));
    }

    csv_t routelist_df("routelist.csv", "r", true, true);
    route_t routes;
    map<string, string> header = {
        {"route", "wrto_route"},
        {"oper", "wrto_oper"},
        {"seq", "wrto_seq_num"},
        {"desc", "wrto_opr_shrt_desc"}
    };
    routelist_df.setHeaders(header);
    routelist_df.trim(" ");
    
    csv_t df;
    vector<vector<string> > data = routelist_df.getData();

    vector<string> routenames = routelist_df.getColumn("route");
    set<string> routelist_set(routenames.begin(), routenames.end()) ;
    routenames = vector<string>(routelist_set.begin(), routelist_set.end());

    iter(routenames, i){
        df = routelist_df.filter("route", routenames[i]);
        routes.setRoute(routenames[i], df);
    }

    iter(alllots, i){
        if(routes.isLotInStations(alllots[i])){ // check if lot is in WB - 7
            lots.push_back(alllots[i]);
        }
    }

    return 0;

}
