#ifndef __JOB_H__
#define __JOB_H__

#include <include/common.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

class lot_t
{
protected:
    std::string _route;
    std::string _lot_number;
    std::string _pin_package;
    std::string _recipe;
    std::string _prod_id;
    std::string _process_id;
    std::string _bom_id;
    std::string _part_id;
    std::string _part_no;

    int _qty;
    int _oper;
    int _lot_size;
    int _amount_of_wires;
    int _amount_of_tools;

    bool _hold;
    bool _mvin;
    bool _is_sub_lot;

    double _queue_time;  // for all queue time;
    double _fcst_time;   // for DA fcst time
    double _outplan_time;

    bool _finish_traversal;

    std::vector<std::string> _log;

public:
    int tmp_oper;
    bool tmp_mvin;


    lot_t(std::map<std::string, std::string> elements) noexcept(false);
    // lot_t(lot_t & lot);
    // lot_t & operator=(lot_t &);
    lot_t() {}
    std::vector<lot_t> createSublots();
    void checkFormation() noexcept(false);

    inline bool isTraversalFinish() { return _finish_traversal; }

    inline int oper() { return _oper; }

    inline std::string route() { return _route; }

    inline bool hold() { return _hold; }

    inline bool mvin() { return _mvin; }

    inline void addLog(std::string _text) { _log.push_back(_text); }

    inline std::string log()
    {
        std::string text;
        iter(_log, i) { text += _log[i] + " "; }
        return text;
    }

    inline std::string lotNumber() { return _lot_number; }

    inline void setTraverseFinished() { _finish_traversal = true; }

    inline std::string recipe() { return _recipe; }

    inline void setProcessId(std::string process_id)
    {
        _process_id = process_id;
    }

    inline std::string processId() { return _process_id; }

    inline std::string prodId() { return _prod_id; }

    inline std::string bomId() { return _bom_id; }

    inline void setBomId(std::string bom_id) { _bom_id = bom_id; }

    inline void setLotSize(int lotsize) noexcept(false)
    {
        if (lotsize < 0)
            throw std::invalid_argument("lot size is less then 0");
        _lot_size = lotsize;
    }

    inline int qty() { return _qty; }

    inline bool isSubLot() { return _is_sub_lot; }

    inline void setFcstTime(double time)
    {
        _fcst_time += time;
        _queue_time += time;
    }

    inline void addQueueTime(double time) { _queue_time += time; }

    inline double fcst() { return _fcst_time; }

    inline double queueTime() { return _queue_time; }

    inline std::string info()
    {
        return "{Lot Number : " + _lot_number + ", route : " + _route +
                   ", recipe : ",
               _recipe + "}";
    }

    inline void setPardId(std::string partid) { _part_id = partid; }

    inline void setPartNo(std::string part_no) { _part_no = part_no; }

    inline std::string part_id() { return _part_id; }

    inline std::string part_no() { return _part_no; }

    inline void setAmountOfTools(int tool) { _amount_of_tools = tool; }

    inline void setAmountOfWires(int amount) { _amount_of_wires = amount; }

    inline int getAmountOfMachines()
    {
        return std::min(_amount_of_tools, _amount_of_wires);
    }
};


#endif
