#ifndef __INFO_H__
#define __INFO_H__

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct __info_t {
    union {
        char text[32];
        unsigned int number[8];
    } data;
    unsigned int text_size : 5;
    unsigned int number_size : 3;
};


bool isSameInfo(struct __info_t info1, struct __info_t info2);

#ifdef __cplusplus
}
#endif

#endif
