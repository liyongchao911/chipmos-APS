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

            /*vector<lot_t> lots =
                createLots("WipOutPlanTime_.csv", "product_find_process_id.csv",
                           "process_find_lot_size_and_entity.csv", "fcst.csv",
                           "routelist.csv", "newqueue_time.csv");
            csv_t out("out.csv", "w");
            iter(lots, i){
                out.addData(lots[i].data());
            }
            out.write();*/
    srand( time(0));
    csv_t time("machine.csv","r",true,true);
    time.trim(" ");
    csv_t location("Location For WB.csv","r",true,true);
    location.trim(" ");
    unsigned int locationrows=location.nrows();

    vector<string>Location=location.getColumn("Location");
    for(int i=0;i<Location.size();i++){
        cout<<Location[i]<<'\n';
    }

    machines_t machines;
    unsigned int nrows = location.nrows();
    map<string, string> elements;
    
    //vector<string>store=time.getRow(1);
    // elements["MODEL"]=store[2];
    // elements["AREA"]=store[21];
    // elements["ENTITY"]=store[0];
    // //cout<<elements["ENTITY"];
    // machines.addMachine(elements);
    



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
