#include <cstdio>
#include <string>
#include <vector>

#define iter(vec, id) for (unsigned int id = 0; id < vec.size(); ++id)


#define iter_range(vec, id, start, end) \
    for (unsigned int id = start; id < end; ++id)

std::vector<std::string> split(char *text, char delimiter);
