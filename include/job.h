#ifndef __JOB_H__
#define __JOB_H__

#include <map>
#include <stdexcept>
#include <vector>
#include <string>
#include <include/common.h>

class lot_t{
protected:
    std::string _route;
    std::string _lot_number;
    std::string _pin_package;
    std::string _recipe;
    std::string _prod_id;
    std::string _process_id;
    std::string _bom_id;
    
    int _qty;
    int _oper;
    unsigned int _lot_size;

    bool _hold;
    bool _mvin;


    double _queue_time; // for all queue time;
    double _fcst_time; // for DA fcst time
    double _outplan_time;

    bool _finish_traversal;

    std::vector<std::string> _log;

public:
    lot_t(std::map<std::string, std::string> elements) noexcept(false);
    lot_t(){};
    int tmp_oper;
    bool tmp_mvin;

    void checkFormation() noexcept(false);

    inline bool isTraversalFinish(){
        return _finish_traversal;
    }

    inline int oper(){
        return _oper;
    }
    
    inline std::string route(){
        return _route;
    }
    
    inline bool hold(){
        return _hold;
    }

    inline bool mvin(){
        return _mvin;
    }

    inline void addLog(std::string _text){
        _log.push_back(_text);
    }

    inline std::string log(){
        std::string text;
        iter(_log, i){
            text += _log[i] + " ";
        }
        return text;
    }

    inline std::string lotNumber(){
        return _lot_number;
    } 

    inline void setTraverseFinished(){
        _finish_traversal = true;
    }

    inline std::string recipe(){
        return _recipe;
    }

    inline void setProcessId(std::string process_id){
        _process_id = process_id; 
    }
    
    inline std::string processId(){
        return _process_id; 
    }

    inline std::string prodId(){
        return _prod_id;
    }

    inline std::string bomId(){
        return _bom_id;
    }

    inline void setBomId(std::string bom_id){
        _bom_id = bom_id;
    }

    inline void setLotSize(int lotsize) noexcept(false){
        if(lotsize < 0)
            throw std::invalid_argument("lot size is less then 0");
        _lot_size = lotsize;
    }
    
};


#endif
