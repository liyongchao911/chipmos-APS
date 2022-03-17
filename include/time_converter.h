#ifndef __TIME_CONVERTER_H__
#define __TIME_CONVERTER_H__

#include <ctime>
#include <regex>
#include <string>
#include <vector>

class time_converter_base_t
{
private:
    std::string pattern_str;
    std::regex pattern;

protected:
    void initialized_tm(struct tm *_tm);
    time_converter_base_t(std::string pattern);

public:
    virtual bool isThisType(std::string text);
    virtual time_t operator()(std::string text) = 0;
};

class time_converter_with_dash_without_second_t : public time_converter_base_t
{
public:
    time_converter_with_dash_without_second_t();
    virtual time_t operator()(std::string text);
};

class time_converter_with_dash_with_second_t : public time_converter_base_t
{
public:
    time_converter_with_dash_with_second_t();
    virtual time_t operator()(std::string text);
};

class time_converter_with_slash_without_second_t : public time_converter_base_t
{
public:
    time_converter_with_slash_without_second_t();
    virtual time_t operator()(std::string text);
};

class time_converter_with_slash_with_second_t : public time_converter_base_t
{
public:
    time_converter_with_slash_with_second_t();
    virtual time_t operator()(std::string text);
};

class time_converter_only_date_with_slash_t : public time_converter_base_t
{
public:
    time_converter_only_date_with_slash_t();
    virtual time_t operator()(std::string text);
};

class time_converter_only_date_with_dash_t : public time_converter_base_t
{
public:
    time_converter_only_date_with_dash_t();
    virtual time_t operator()(std::string text);
};



class timeConverter
{
private:
    static std::vector<time_converter_base_t *> converters;

    time_t _base_time;

public:
    inline timeConverter() : _base_time(0) {}
    inline timeConverter(std::string text) : _base_time(0)
    {
        _base_time = this->operator()(text);
    }
    inline timeConverter(time_t base_time) : _base_time(base_time) {}
    inline time_t getBaseTime() { return _base_time; }
    inline void setBaseTime(std::string text)
    {
        _base_time = 0;
        _base_time = this->operator()(text);
    }
    time_t operator()(std::string text);
};

#endif
