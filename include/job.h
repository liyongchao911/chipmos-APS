#ifndef __JOB_H__
#define __JOB_H__

#include <map>
#include <vector>
#include <string>

class lot_t{
protected:
    std::string _route;
    std::string _lot_number;
    std::string _pin_package;
    int _qty;
    int _oper;
    bool _hold;
    bool _mvin;

public:
    lot_t(std::map<std::string, std::string> elements);
    
    int oper();
    std::string route();

};

#endif
