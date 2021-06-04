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


using namespace std;

bool testIsTheSame(map<string, map<int, card_t> > models1, string src1, map<string, map<int, card_t> > models2, string src2);


int main(int argc, const char *argv[])
{
    vector<lot_t> lots =
        createLots("WipOutPlanTime_.csv", "product_find_process_id.csv",
                   "process_find_lot_size_and_entity.csv", "fcst.csv",
                   "routelist.csv", "newqueue_time.csv");
    return 0;
    // condition_cards_h cards(12, "UTC-1000", "UTC-1000S", "UTC-2000",
    //                         "UTC-2000S", "UTC-3000", "UTC-5000S", "Maxum Base",
    //                         "Maxum Plus", "Maxum Ultra", "Iconn", "Iconn Plus",
    //                         "RAPID");
    // cards.addMapping("Maxum (Ultra)", 2, "Maxum", "Maxum Ultra");
    // cards.addMapping("Ultra", 1, "Maxum-Ultra");
    // cards.addMapping("Plus", 1, "Maxum-Plus");

    // cards.readConditionCardsDir("./WB/CARD_OFFICAL");
    // cards.readConditionCardsDir("./WB/CARD_TEMP");

    
    // printf("--------------------------------------------------------\n");
    // testIsTheSame(models, "bd_id_mapping", card_models, "cards_models");
    
    return 0;
}



bool testIsTheSame(map<string, map<int, card_t> > models1, string src1, map<string, map<int, card_t> > models2, string src2){
    card_t card1;
    card_t card2;
    bool retval = true;
    string bd_id;
    int oper;
    
    for(map<string, map<int, card_t> > ::iterator it = models1.begin(); it != models1.end(); it++){
        bd_id = it->first;
        for(map<int, card_t>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++){
            oper = it2->first;
            try{
                card1 = models1.at(bd_id).at(oper);
            }catch(std::out_of_range & e){
                printf("%s doesn't have (%s, %d)\n", src1.c_str(), bd_id.c_str(), oper);
                continue;
            }
            try{
                card2 = models2.at(bd_id).at(oper);
            }catch(std::out_of_range & e){
                printf("%s doesn't have (%s, %d)\n", src2.c_str(), bd_id.c_str(), oper);
                continue;
            }
            set<string> s1(card1.models.begin(), card1.models.end());
            set<string> s2(card2.models.begin(), card2.models.end());
            if(s1 != s2){
                printf("%s-%d is incorrect, ", bd_id.c_str(), oper);    
                vector<string> m1 = card1.models;
                vector<string> m2 = card2.models;
                string s_model1 = join(m1, "/");
                string s_model2 = join(m2, "/");
                printf("%s, ", s_model1.c_str());
                printf("%s\n", s_model2.c_str());
                retval = false;
            }
        }
    }
    return true;
}
