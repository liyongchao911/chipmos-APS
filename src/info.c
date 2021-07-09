#include "include/info.h"

bool isSameInfo(struct __info_t info1, struct __info_t info2)
{
    if (info1.number_size != info2.number_size) {
        return false;
    } else {
        for (unsigned int i = 0; i < info1.number_size; i++)
            if (info1.data.number[i] != info2.data.number[i])
                return false;
    }
    return true;
}
