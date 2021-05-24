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
    int year, month, day, hour, minute, second;// 定义时间的各个int临时变量。
        sscanf(text, "%d/%d/%d %d:%d", &month, &day, &year, &hour, &minute);// 将string存储的日期时间，转换为int临时变量。
        tm_.tm_year = year+100;                 // 年，由于tm结构体存储的是从1900年开始的时间，所以tm_year为int临时变量减去1900。
        tm_.tm_mon = month - 1;                    // 月，由于tm结构体的月份存储范围为0-11，所以tm_mon为int临时变量减去1。
        tm_.tm_mday = day;                         // 日。
        tm_.tm_hour = hour;                        // 时。
        tm_.tm_min = minute;                       // 分。
        tm_.tm_isdst = 0;                          // 非夏令时。
        time = mktime(&tm_);

        //printf("1");

    // TODO : convert text to time;
    //

    return time;
}

