#ifndef __INFRA_H__
#define __INFRA_H__

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "include/info.h"

#define iter(vec, id) for (unsigned int id = 0; id < vec.size(); ++id)

#define iter_range(vec, id, start, end) \
    for (unsigned int id = start; id < end; ++id)

/**
 * stringify
 */
#define _str(x) #x
#define xstr(x) _str(x)

std::vector<std::string> split(char *text, char delimiter);

std::string join(std::vector<std::string> strings, std::string delimiter);

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

/**
 * timeConverter() - convert text to time_t
 */
time_t timeConverter(std::string text);

void stringToLower(char *text);
void stringToUpper(char *text);


struct __info_t stringToInfo(std::string s);


void random(double *genes, int size);

int randomRange(int start, int end, int different_num);

double randomDouble();


#endif
