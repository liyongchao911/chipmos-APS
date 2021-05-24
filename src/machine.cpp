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
    std::string location = _entity_location[elements["ENTITY"]];
    _entities[elements["MODELS"]][location].push_back( entity_t{
            .recover_time = 0,
            .entity_name=elements["ENTITY"],
            .model_name = elements["MODELS"],
            .area=elements["AREA"]

});


}
machines_t::machines_t(const char* text){
    double t1=timeConverter(text);
    double t2=timeConverter("12/19/20 10:50");

    settime(t2-t1);

}
//int main(){

//}
