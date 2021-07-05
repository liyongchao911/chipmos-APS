#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "include/lot.h"
#include "include/csv.h"
#include "include/route.h"

using namespace std;

int main(){
    csv_t route_list_csv("routelist.csv", "r", true, true);
    csv_t queue_time("newqueue_time.csv", "r", true, true);  
    route_list_csv.trim(" ");
    route_list_csv.setHeaders(
            map<string, string>({
                {"route", "wrto_route"},
                {"oper", "wrto_oper"},
                {"seq", "wrto_seq_num"},
                {"desc", "wrto_opr_shrt_desc"}
                                })
            );


    route_t routes;
    routes.setQueueTime(queue_time);
    routes.setRoute(route_list_csv);



    return 0;
}


