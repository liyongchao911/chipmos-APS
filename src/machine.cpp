#include <include/csv.h>
#include <time.h>
#include <map>
#include <string>
#include <vector>
#include <include/machine.h>

void machines_t::addMachine(std::map<std::string, std::string> elements){

    
    csv_t LFW("Location For WB.csv","r",true,true);
    std::vector <std::string>Location=LFW.getColumn("Location");
    std::vector <std::string>LFWEntity=LFW.getColumn("Entity");
    for(int i=0;i<Location.size();i++){
       _entity_location[LFWEntity.at(i)]=Location.at(i); 
    }
    if( _entity_location.empty() ) {
        return;
    }
    std::string location = _entity_location[elements["ENTITY"]];
    _entities[elements["MODEL"]][location].push_back( entity_t{
            .recover_time = 0,
            .entity_name=elements["ENTITY"],
            .model_name = elements["MODEL"],
            .area=elements["AREA"]});
    //std::cout<<_entities[elements["MODEL"]][location][0].area;
}
void machines_t::addMachines(csv_t LFW,csv_t WB){
    std::vector <std::string>Location=LFW.getColumn("Location");
    std::vector <std::string>LFWEntity=LFW.getColumn("Entity");
    std::vector<std::string>Area=WB.getColumn("AREA");
    std::vector<std::string>Model=WB.getColumn("MODEL");
    std::vector<std::string>WBEntity=WB.getColumn("ENTITY");
    unsigned int WBrows =Model.size();
    unsigned int LFWrows =Location.size();
    for(int i=0;i<WBrows;i++){
        for(int j=0;j<LFWrows;j++){
            if(LFWEntity.at(j)==WBEntity.at(i)){
                _entities[Model.at(i)][Location.at(j)].push_back( entity_t{
                .recover_time = 0,
                .entity_name=LFWEntity[j],
                .model_name = Model[i],
                .area=Area[i]
                 });
                 break;
            }

        }

    }


}
std::vector<entity_t> machines_t::randomlyGetEntities(std::string model_name,
                                          std::string area,
                                          int amount){
    std::vector<entity_t>store;
    int size=_entities[model_name][area].size();
    bool isUse[size];
    for(int i=0;i<amount;i++){

        int number=rand()%size;
        if(isUse[number]==false){
           store.push_back(_entities[model_name][area][number]);
           isUse[number]=true;

           }
        else{
            i--;
        }

    }


    return store;
}
machines_t::machines_t(const char* text){

    double t1;
    // t1=(double)timeConverter(text);
    double t2;
    // t2=(double)timeConverter("12/19/20 10:50");


    this->time = t2 - t1;

}

