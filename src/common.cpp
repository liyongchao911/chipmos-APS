#include <include/common.h>


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
