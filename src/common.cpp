#include <include/common.h>

using namespace std;
std::vector<std::string> split(char *text, char delimiter)
{
    char *iter = text, *prev = text;
    std::vector<std::string> data;
    while (*iter) {
        if (*iter == delimiter ||
            *iter == '\n') {  // for unix-like newline character
            *iter = '\0';
            data.push_back(prev);
            prev = ++iter;
        } else if (*iter == '\r' &&
                   *(iter + 1) ==
                       '\n') {  // for windows newline characters '\r\n'
            *iter = '\0';
            data.push_back(prev);
            iter += 2;
            prev = iter;
        } else if (*(iter + 1) == '\0') {
            data.push_back(prev);
            ++iter;
        } else {
            ++iter;
        }
    }

    return data;
}

std::string join(std::vector<std::string> strings, std::string delimiter)
{
    if (strings.size() == 0)
        return "";
    std::string s;
    iter_range(strings, i, 0, strings.size() - 1)
    {
        s += strings[i];
        s += delimiter;
    }
    s += strings[strings.size() - 1];
    return s;
}

void stringToLower(char *text)
{
    for (; *text; ++text)
        *text |= 0x20;
}

void stringToUpper(char *text)
{
    for (; *text; ++text)
        *text ^= 0x20;
}

time_t timeConverter(const char *text)
{
    time_t time;

    tm tm_;
    int year, month, day, hour, minute, second;
        sscanf(text, "%d/%d/%d %d:%d", &month, &day, &year, &hour, &minute);
        tm_.tm_year = year+100;                 // only last two digit ,so plus 100
        tm_.tm_mon = month - 1;                    // monte:tm structure store 0-11 so minus one
        tm_.tm_mday = day;                         // day
        tm_.tm_hour = hour;                        // hour
        tm_.tm_min = minute;                       // minute
        tm_.tm_sec =0;                             //second
        tm_.tm_isdst = 0;                          // whether summer time or not
        time = mktime(&tm_);

        //printf("1");

    // TODO : convert text to time;
    //

    return time;
}

