#ifndef __MACHINE_H__
#define __MACHINE_H__

#include <include/machine_base.h>
#include <include/infra.h>

typedef struct __info_t machine_info_t;
typedef struct __info_t tool_info_t;
typedef struct __info_t wire_info_t;

typedef struct __tool_t{
    tool_info_t name;
    time_t time;
}tool_t;

typedef struct __wire_t{
    wire_info_t name;
    time_t time;
}wire_t;

typedef struct __machine_t{
    machine_base_t base;
    tool_t * tool;
    wire_t * wire;
}machine_t;

#endif
