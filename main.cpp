#include <ctime>
#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include <include/arrival.h>
#include <include/common.h>
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
    csv_t area("Location For WB.csv","r",true,true);
    area.trim(" ");

    csv_t entity;
    entity = time.filter("ENTITY", "BB211");

    //cout<<entity;
    machines_t machines;
    unsigned int nrows = time.nrows();
    for(unsigned int i = 0; i < nrows; ++i){
        machines.addMachine(time.getElements(i));
    }

    //cout<<machine
    //_entity["UTC-3000"]["TA-3"].recovertime=timeConverter(char *text);
    vector<string> outplan = time.getColumn("OUTPLAN");
    iter(outplan, i){

        cout<<timeConverter(outplan[i].c_str())<<endl;
    }

   return 0;
}
