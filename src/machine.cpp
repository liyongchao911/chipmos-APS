#ifndef __MACHINE_H__
#define __MACHINE_H__

#include <include/csv.h>
#include <time.h>
#include <map>
#include <string>
#include <vector>
#include <include/machine.h>
//int main(){
//    _entity["UTC-1000"][
//}

void machines_t::addMachine(std::map<std::string, std::string> elements){
    _entities[elements["MODELS"]][elements["AREA"]] = entity_t{
            .recover_time = 0,
            .model_name = elements["MODELS"],
            .entity_name=elements["ENTITY"],
            .area=elements["AREA"]

}


}
//int main(){
//    cout<<elements["MODELS"];
//}
