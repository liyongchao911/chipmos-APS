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
#include <include/common.h>
#include <include/condition_card.h>
#include <include/csv.h>
#include <include/da.h>
#include <include/job.h>
#include <include/route.h>
#include <include/machine.h>


using namespace std;



int main(int argc, const char *argv[])
{
//    vector<lot_t> lots =
//        createLots("WipOutPlanTime_.csv", "product_find_process_id.csv",
//                   "process_find_lot_size_and_entity.csv", "fcst.csv",
//                   "routelist.csv", "newqueue_time.csv");
    csv_t time("machine.csv","r",true,true);
    time.trim(" ");
    csv_t location("Location For WB.csv","r",true,true);
    location.trim(" ");
    unsigned int locationrows=location.nrows();

    vector<string>Location=location.getColumn("Location");
//    for(int i=0;i<Location.size();i++){
//        cout<<Location[i]<<'\n';
//    }

    //csv_t entity;
    //entity = time.filter("ENTITY", "BB211");

    //cout<<entity;
    machines_t machines;
    unsigned int nrows = location.nrows();
    map<string, string> _entity_location;

    /** MAP: _entity_location["ENTITY"]=Location*/
    for(unsigned int i = 0; i < nrows; ++i){
        _entity_location[location.getElement("ENTITY",i)]=location.getElement("Location",i);

    }
    machines.addMachine(_entity_location);
//    cout<<machines.time;
    //cout<<machine
    //_entity["UTC-3000"]["TA-3"].recovertime=timeConverter(char *text);
//    vector<string> outplan = time.getColumn("OUTPLAN");
//    iter(outplan, i){

//        cout<<timeConverter(outplan[i].c_str())<<endl;
//    }
    machines_t x("12/19/20 10:30");
    cout<<x.gettime();
   return 0;
}
