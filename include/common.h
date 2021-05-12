#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstdio>
#include <string>
#include <vector>

#define iter(vec, id) for (unsigned int id = 0; id < vec.size(); ++id)


#define iter_range(vec, id, start, end) \
    for (unsigned int id = start; id < end; ++id)

std::vector<std::string> split(char *text, char delimiter);

template <class T>
std::vector<T> operator+(std::vector<T> &op1, std::vector<T> op2)
{
    std::vector<T> result(op1.begin(), op1.end());
    iter(op2, i) { result.push_back(op2[i]); }
    return result;
}

template <class T>
std::vector<T> operator+=(std::vector<T> &op1, std::vector<T> op2)
{
    iter(op2, i) { op1.push_back(op2[i]); }
    return op1;
}
#endif
