#ifndef __CONDITION_CARD_H__
#define __CONDITION_CARD_H__

#include <include/common.h>
#include <include/csv.h>
#include <stdarg.h>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

typedef struct {
    unsigned int oper;
    std::string recipe;
    std::vector<std::string> models;
} card_t;

bool isCsvFile(std::string filename);
bool isExcelFile(std::string filename);


class condition_cards_h
{
private:
    // recipe -> oper -> card
    std::map<std::string, std::map<int, card_t> > _models;

    std::set<std::string> _model_set;
    std::map<std::string, std::vector<std::string> > _model_mapping;

    std::string dirName(std::string dir);

    std::vector<std::string> _log;

protected:
    virtual std::string dataScrubbing(std::string text);
    virtual char *dataScrubbing(char *);
    virtual inline void addLog(std::string text) { _log.push_back(text); }

public:
    condition_cards_h();

    condition_cards_h(int n, ...);

    void addMapping(std::string std_model_name, int n, ...);

    void readConditionCardsDir(std::string dir_name, bool replace = false);

    void readConditionCards(std::string sub_dir_name, bool replace = false);

    std::vector<std::string> log();

    card_t readConditionCard(std::string filename,
                             std::string recipe = "",
                             unsigned int oper = 0);

    inline card_t getModels(std::string recipe, int oper)
    {
        return _models[recipe][oper];
    }
};

#endif
