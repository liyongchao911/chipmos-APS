#ifndef __INFO_H__
#define __INFO_H__

#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

struct __info_t {
    union {
        char text[64];
        unsigned long long number[8];
    } data;
    unsigned int text_size : 6;
    unsigned int number_size : 4;
};

typedef struct __info_t info_t;


bool isSameInfo(struct __info_t info1, struct __info_t info2);

info_t emptyInfo();

#ifdef __cplusplus
}
#endif

#endif
