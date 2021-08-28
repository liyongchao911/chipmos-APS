#ifndef __LOT_H__
#define __LOT_H__

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/csv.h"
#include "include/infra.h"
#include "include/job.h"

#define ERROR_TABLE                                                       \
    X(SUCCESS, = 0x00, "SUCCESS")                                         \
    X(ERROR_WIP_INFORMATION_LOSS, = 0x01, "ERROR_WIP_INFORMATION_LOSS")   \
    X(ERROR_PROCESS_ID, = 0x02, "ERROR_PROCESS_ID")                       \
    X(ERROR_BOM_ID, = 0x03, "ERROR_BOM_ID")                               \
    X(ERROR_LOT_SIZE, = 0x04, "ERROR_LOT_SIZE")                           \
    X(ERROR_INVALID_LOT_SIZE, = 0x05, "ERROR_INVALID_LOT_SIZE")           \
    X(ERROR_DA_FCST_VALUE, = 0x06, "ERROR_DA_FCST_VALUE")                 \
    X(ERROR_INVALID_OPER_IN_ROUTE, = 0x07, "ERROR_INVALID_OPER_IN_ROUTE") \
    X(ERROR_INVALID_QUEUE_TIME_COMBINATION, = 0x08,                       \
      "ERROR_INVALID_QUEUE_TIME_COMBINATION")                             \
    X(ERROR_HOLD, = 0x09, "ERROR_HOLD")                                   \
    X(ERROR_WB7, = 0x0A, "ERROR_WB7")                                     \
    X(ERROR_CONDITION_CARD, = 0x0B, "ERROR_CONDITION_CARD")               \
    X(ERROR_PART_ID, = 0x0C, "ERROR_PART_ID")                             \
    X(ERROR_NO_WIRE, = 0x0D, "ERROR_NO_WIRE")                             \
    X(ERROR_WIRE_MAPPING_ERROR, = 0x0E, "ERROR_WIRE_MAPPING_ERROR")       \
    X(ERROR_PART_NO, = 0x0F, "ERROR_PART_NO")                             \
    X(ERROR_NO_TOOL, = 0x10, "ERROR_NO_TOOL")                             \
    X(ERROR_TOOL_MAPPING_ERROR, = 0x11, "ERROR_TOOL_MAPPING_ERROR")       \
    X(ERROR_UPH_FILE_ERROR, = 0x12, "ERROR_UPH_FILE")                     \
    X(ERROR_UPH_0, = 0x13, "ERROR_UPH_0")


#define X(item, value, name) item value,
enum ERROR_T { ERROR_TABLE };
#undef X

extern const char *ERROR_NAMES[];

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
    std::string _urgent;
    std::string _customer;
    std::string _wb_location;
    std::string _prescheduled_machine;
    std::string _prescheduled_model;

    int _qty;
    int _oper;
    int _lot_size;
    int _sub_lots;
    int _amount_of_wires;
    int _amount_of_tools;
    int _prescheduled_order;

    bool _hold;
    bool _mvin;
    bool _is_sub_lot;

    double _queue_time;  // for all queue time;
    double _fcst_time;   // for DA fcst time
    double _outplan_time;
    double _weight;

    bool _finish_traversal;

    std::vector<std::string> _log;
    std::vector<std::string> _can_run_models;
    std::vector<std::string> _can_run_locations;
    std::vector<std::string> _can_run_entities;

    std::map<std::string, double> _uphs;
    std::map<std::string, double> _model_process_times;
    std::map<std::string, double> _entity_process_times;

    enum ERROR_T _status;

public:
    int tmp_oper;
    bool tmp_mvin;

    lot_t();

    /**
     * constructor for lot_t
     *
     * This constructor for lot_t has only one parameter whose type is
     * std::map<std::string, std::string>. The parameter is used to store the
     * relationship between key and data from dataframe. For example,
     * elements["route"] == "BGA321", elements["lot_number"] == "AASJSDKA01".
     * The mapping relationship is from WIP(work in process) sheet. The
     * constructor will initialize lots of data members. After initalizing the
     * data members, the constructor will checke if data member has incorrect
     * data. If data has incorrect data, the constructor will throw exception
     * and the error message will point out which information is not provided or
     * incorrect.
     *
     * @exception : std::invalid_argument
     * @param elements : mapping relationship between key and data for
     * datamember.
     */
    lot_t(std::map<std::string, std::string> elements) noexcept(false);

    /**
     * createSublots() - create sub lots
     *
     * The function is used to creat sub-lot, if lot isn't sub-lot. The number
     * of sublots is ⌈(qty / lot_size)⌉.
     *
     * @return a vector of lot_t object splited from the lot.
     */
    std::vector<lot_t> createSublots();

    /**
     * checkFormation() - check the basic formation of object is correct or not.
     * checkFormation is used to check the basic fomation of data member is
     * correct or not. Basic fomation of data member is from WIP(work in
     * process) sheet. If basic formation is incorrect or not provided, the lot
     * will not be involved in scheduling plan.
     *
     * @exception : std::invalid_argumaent
     */
    bool checkFormation();

    /**
     * addLog () - add log for this lot
     */
    void addLog(std::string _text, enum ERROR_T code);

    /**
     * addQueueTime () - add queue time
     *
     * The unit of queue time is minute. Lot initialize _queue_time as 0 in the
     * constructor. addQueueTime is used to add time to _queue_time. The
     * function is used in route_t::calculateQueueTime to add queue time station
     * by station.
     *
     * @param time : queue time in a station
     */
    void addQueueTime(double time);

    /**
     * setTraverseFinished () - set the traversing flag to be finished
     *
     * If the lot finish traversing the route, the traversing flag need to be
     * set by using this function. The lot will no longer traverse the route
     * again.
     */
    void setTraverseFinished();

    /**
     * setProcessId () - setup the process id
     *
     * The process id is mapped by product_id.
     *
     * @param a std::string type parameter
     */
    void setProcessId(std::string process_id);

    /**
     * setBomId () - setup Bom Id
     *
     * Bom id is mapped by process id
     *
     * @param a std::string type parameter
     */
    void setBomId(std::string bom_id);

    /**
     * setLotSize () - setup lot size
     *
     * The lot can be split out to several sub-lots. The number of qty of each
     * sublot is determined by lot size.
     *
     * @exception : std::invalid_argument if parameter less than 0
     * @param lotsize : a integer type of parameter
     */
    void setLotSize(int lotsize) noexcept(false);

    /**
     * setFcstTime () - setup forecast time
     *
     * forcast time is used to predict the queue time in D/A station
     *
     * @param time : a double type of variable which is in minute unit
     */
    void setFcstTime(double time);

    /**
     * setPartId () - setup part_id
     *
     * part id is used to determine what kind of wire should be used for this
     * lot
     *
     * @param partid : a std::string type of variable.
     */
    void setPartId(std::string partid);

    /**
     * setPartNo () - setup part_no
     *
     * part_no is used to determine what kind of tool should be used for this
     * lot
     *
     * @param part_no : a std::string type of variable
     */
    void setPartNo(std::string part_no);

    /**
     * setAmountOfTools () - setup amount of tools can be used for this lot
     *
     * @param amount : a integer type of variable
     */
    void setAmountOfTools(int amount);

    /**
     * setAmountOfWires () - setup amount of wires which can be used for this
     * lot
     *
     * @param amount : a integer type of variable
     */
    void setAmountOfWires(int amount);

    /**
     * setCanRunModel () - setup a single model
     */
    void setCanRunModel(std::string model);

    /**
     * setCanRunModels () - setup multiple models
     */
    void setCanRunModels(std::vector<std::string> models);

    /**
     * getAmountOfTools () - return the number of available tools for this lot
     * @return  integer
     */
    int getAmountOfTools();

    /**
     * getAmountOfWires () - return the number of available wires for this lot
     * @return
     */
    int getAmountOfWires();

    /**
     * getCanRunModels () - return a vector of string which represents can run
     * models
     * @return
     */
    std::vector<std::string> getCanRunModels();

    /**
     * oper () - current oper of this lot
     */
    int oper();

    /**
     * qty () - quantity of this lot
     */
    int qty();

    /**
     * hold () - Is this lot held or not
     *
     * @return true if lot is held
     */
    bool hold();

    /**
     * mvin () - Is this lot move into the machine or not
     *
     * @return true if lot is moving into the machine
     */
    bool mvin();

    /**
     * isSubLot () - Is this lot a sub-lot
     *
     * if length of lot number greater then 8 which means that the lot isn't a
     * sub-lot
     *
     * @return true if lot is a sub-lot
     */
    bool isSubLot();

    /**
     * isTraversalFinished () - Is the traversal finished or not?
     *

     * Lot will traverse each station in the route to sum the queue time and
     determine the arrival time to W/B station. isTraversalFinish is going to
     describe that if traversal is finished or not.
     *
     * @return true if traversal is finished.
     */
    bool isTraversalFinish();

    /**
     * route () - get the route of this lot
     *
     * @return std::string type data
     */
    std::string route();

    /**
     * log () - get the log of lot
     *
     * @return std::string type data
     */
    std::string log();

    /**
     * lotNumber () - get the lot_number
     *
     * lotNumber in WIP(work in process) is unique which represent an identifier
     * of a lot.
     *
     * @return std::string type data
     */
    std::string lotNumber();

    /**
     * recipe () - get the recipe of the lot
     *
     * recipe is bd_id.
     *
     * @return std::string type data
     */
    std::string recipe();

    /**
     * processId () - get the process_id of the lot
     *
     * process id is mapped by prod_id. Initially, process_id is empty because
     * this information is not in WIP. process_id is set by setProcessId.
     *
     * @return std::string type data
     */
    std::string processId();

    /**
     * prodId () - get the product_id of the lot
     *
     * product id is in WIP so that the return data will not be an empty string.
     *
     * @return std::string type data
     */
    std::string prodId();

    /**
     * bomId () - get the bom_id of the lot
     *
     * bom id, mapped by process_id, is used to determine what kind of wire
     * should be used on this lot.  Initially, bom_id is an empty string. Call
     * setBomId to setup the bom_id
     *
     * @return std::string type data
     */
    std::string bomId();

    /**
     * pin_package () - get the pin_package of the lot
     *
     * pin_package, an information of lot, is used to
     */
    std::string pin_package();

    /**
     * info () - get the infomation of the lot
     *
     * The infomation includes lot_number, route and recipe. Note that info is a
     * virtual function.
     *
     * @return std::string type data which include lot_number, route and recipe
     */
    virtual std::string info();

    /**
     * part_id () - get the part id of the lot
     *
     * @return std::string type of data.
     */
    std::string part_id();

    /**
     * part_no () - get the part number of the lot
     *
     * @return std::string type of data.
     */
    std::string part_no();

    /**
     * fcst () - get the forecast time of the lot
     *
     * @return forecast time in DA station, double data type, in minutes.
     */
    double fcst();

    /**
     * queueTime () - get the queue time of the lot
     *
     * if lot is finished traversing the route, queue time is also the arrival
     * time to W/B station.
     *
     * @return  time, double data type, in minutes
     */
    double queueTime();

    /**
     * data () - return all attribute of this lot
     * The function can be used to output the information of this lot to csv
     * file. The return value is a map container mapping string, attribute name,
     * to another string, attribute value.
     * @return map<string, string> type data
     */
    std::map<std::string, std::string> data();

    /**
     * setUph () - set the uph for specific model
     *  The function is used to set the uph for specific model. If the uph is 0,
     *  the model will be removed because uph==0 is illegal.
     * @param name :  the model name
     * @param uph : double type value
     * @return true if uph isn't 0, otherwise, return false.
     */
    bool setUph(std::string name, double uph);

    /**
     * setUph () - set the uph by dataframe
     * In this function, correct uph is filtered by recipe, oper, and cust. The
     * function will call setUph(std::string, double) to setup uph for each
     * model.
     * @param uph : csv_t type dataframe
     * @return true if setup uph successfully, return false if there is no
     * model's uph is set.
     */
    bool setUph(csv_t &uph);

    /**
     * setCanRunLocation () - set the can run location
     * This function will setup the lot's can run location. The can run location
     * is determined by its pin package.
     * @param model_locations : model_locations is a map container mapping
     * model's name to it's locations.
     */
    void setCanRunLocation(
        std::map<std::string, std::vector<std::string> > model_locations);

    /**
     * getCanRunLocation () - get the can run locations of lot
     * @return a vector of string which represents a location
     */
    std::vector<std::string> getCanRunLocations();

    /**
     * isEntitySuitable () - check if the lot can run on this model and location
     * @param location : location of this entity.
     * @return
     */
    bool isEntitySuitable(std::string location);

    /**
     * getCanRunEntities () - get can run entities vector
     * @return
     */
    std::vector<std::string> getCanRunEntities();

    /**
     * job () - generate job_t instance by lot
     * In this function, job_t fields are initialized to its variable. The field
     * includes list.getValue, part_no, pin_package, job_info, customer,
     * part_id, bdid, urgent_code qty, start_time, end_time, arriv_t
     * @return job_t instance
     */
    job_t job();

    /**
     * getEntityProcessTime () - get can run entity and their process time.
     * @return map<string, double> type data mapping entity's name to its
     * process time
     */
    std::map<std::string, double> getEntityProcessTime();

    /**
     * clearCanRunLocation () clear all can run locations
     */
    void clearCanRunLocation();

    /**
     * getUphs () - get all models' uph
     * @return map<string, double> data mapping model's name to its uph
     */
    std::map<std::string, double> getUphs();

    bool isModelValid(std::string model);

    std::string preScheduledEntity();

    bool isPrescheduled();

    int prescheduledOrder();

    void setPrescheduledModel(std::string model);

    void setNotPrescheduled();

    std::map<std::string, double> getModelProcessTimes();
};

inline std::map<std::string, double> lot_t::getModelProcessTimes()
{
    return _model_process_times;
}

inline void lot_t::setNotPrescheduled()
{
    _prescheduled_order = -1;
}

inline void lot_t::setPrescheduledModel(std::string model)
{
    // FIXME : Is it necessary to check whether the lot is preshceduled or not?
    _prescheduled_model = model;
}

inline void lot_t::clearCanRunLocation()
{
    _can_run_locations.clear();
}

inline std::map<std::string, double> lot_t::getUphs()
{
    return _uphs;
}

inline std::vector<std::string> lot_t::getCanRunEntities()
{
    return _can_run_entities;
}

inline std::vector<std::string> lot_t::getCanRunLocations()
{
    return _can_run_locations;
}

inline void lot_t::setBomId(std::string bom_id)
{
    _bom_id = bom_id;
}

inline std::string lot_t::part_no()
{
    return _part_no;
}

inline bool lot_t::isTraversalFinish()
{
    return _finish_traversal;
}

inline int lot_t::oper()
{
    return _oper;
}

// inline int lot_t::getAmountOfMachines()
// {
//     return std::min(_amount_of_tools, _amount_of_wires);
// }
inline int lot_t::getAmountOfTools()
{
    return _amount_of_tools;
}

inline int lot_t::getAmountOfWires()
{
    return _amount_of_wires;
}

inline std::string lot_t::route()
{
    return _route;
}


inline bool lot_t::hold()
{
    return _hold;
}

inline bool lot_t::mvin()
{
    return _mvin;
}

inline void lot_t::addLog(std::string _text, enum ERROR_T code)
{
    _log.push_back(_text);
    if (_status != SUCCESS) {
        fprintf(stderr, "Warning: You set the unscess code to another code!\n");
        _status = code;
    } else
        _status = code;
}

inline std::string lot_t::log()
{
    std::string text;
    iter(_log, i) { text += _log[i] + " "; }
    return text;
}

inline std::string lot_t::lotNumber()
{
    return _lot_number;
}


inline void lot_t::setTraverseFinished()
{
    _finish_traversal = true;
}


inline std::string lot_t::recipe()
{
    return _recipe;
}

inline void lot_t::setProcessId(std::string process_id)
{
    _process_id = process_id;
}

inline std::string lot_t::processId()
{
    return _process_id;
}

inline std::string lot_t::prodId()
{
    return _prod_id;
}

inline std::string lot_t::bomId()
{
    return _bom_id;
}


inline void lot_t::setLotSize(int lotsize) noexcept(false)
{
    if (lotsize < 0)
        throw std::invalid_argument("lot size is less then 0");
    _lot_size = lotsize;
}

inline int lot_t::qty()
{
    return _qty;
}

inline bool lot_t::isSubLot()
{
    return _is_sub_lot;
}

inline void lot_t::setFcstTime(double time)
{
    _fcst_time += time;
    _queue_time += time;
}


inline void lot_t::addQueueTime(double time)
{
    _queue_time += time;
}


inline double lot_t::fcst()
{
    return _fcst_time;
}


inline double lot_t::queueTime()
{
    return _queue_time;
}

inline std::string lot_t::info()
{
    return "{Lot Number : " + _lot_number + ", route : " + _route +
           ", recipe : " + _recipe + "}";
}


inline void lot_t::setPartId(std::string partid)
{
    _part_id = partid;
}


inline void lot_t::setPartNo(std::string part_no)
{
    _part_no = part_no;
}


inline std::string lot_t::part_id()
{
    return _part_id;
}


inline void lot_t::setAmountOfTools(int tool)
{
    _amount_of_tools = tool;
}


inline void lot_t::setAmountOfWires(int amount)
{
    _amount_of_wires = amount;
}

inline bool lot_t::isModelValid(std::string model)
{
    if (_part_no.find("A0801") !=
        std::string::npos) {  // if part_no contains A0801
        if (model.compare("UTC1000") == 0 || model.compare("UTC1000S") == 0 ||
            model.compare("UTC2000") == 0 || model.compare("UTC2000S") == 0 ||
            model.compare("UTC3000") == 0) {
            return true;
        } else
            return false;
    } else if (_part_no.find("A0803") !=
               std::string::npos) {  // if part_no contains A0803
        if (model.compare("UTC1000") != 0 || model.compare("UTC1000S") != 0 ||
            model.compare("UTC2000") != 0 || model.compare("UTC2000S") != 0 ||
            model.compare("UTC3000") != 0) {
            return true;
        } else
            return false;
    }
    return true;
}

inline void lot_t::setCanRunModel(std::string model)
{
    if (isModelValid(model)) {
        _uphs[model] = 0;
        _model_process_times[model] = 0;
        if (find(_can_run_models.begin(), _can_run_models.end(), model) !=
            _can_run_models.end())
            _can_run_models.push_back(model);
    }
}

inline void lot_t::setCanRunModels(std::vector<std::string> models)
{
    iter(models, i)
    {
        if (isModelValid(models[i])) {
            _uphs[models[i]] = 0;
            _model_process_times[models[i]] = 0;
        }
    }

    _can_run_models = models;
}

inline bool lot_t::setUph(std::string model, double uph)
{
    if (uph == 0) {
        _uphs.erase(model);
        _model_process_times.erase(model);
        return false;
    } else {
        _uphs.at(model) = uph;
        _model_process_times[model] = (_qty / uph) * 60;
    }
    return true;
}

inline std::vector<std::string> lot_t::getCanRunModels()
{
    return _can_run_models;
}

inline std::string lot_t::pin_package()
{
    return _pin_package;
}

inline std::string lot_t::preScheduledEntity()
{
    return _prescheduled_machine;
}

inline bool lot_t::isPrescheduled()
{
    return _prescheduled_order >= 0;
}

inline int lot_t::prescheduledOrder()
{
    return _prescheduled_order;
}


typedef struct {
    std::string wire_tools_name;
    std::string wire_name;
    std::string tool_name;
    unsigned long long lot_amount;
    int tool_amount;
    int wire_amount;
    int machine_amount;
    std::map<std::string, int> models_statistic;
    std::map<std::string, int> bdid_statistic;
    // std::vector<entity_t *> entities;
    std::vector<std::string> entity_names;
    std::vector<lot_t *> lots;
} lot_group_t;

/**
 * lotGroupCmp () - compare two group by its lot amount
 * @param g1 : group 1
 * @param g2 : group 2
 * @return true if g1.lot_amount > g2.lot_amount
 */
bool lotGroupCmp(lot_group_t g1, lot_group_t g2);



#endif
