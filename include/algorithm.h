//
// Created by eugene on 2021/7/5.
//

#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#include "include/lots.h"
#include "include/machines.h"

void prescheduling(machines_t *machines, lots_t *lots);

void stage2Scheduling(machines_t *machines, lots_t *lots);

void stage3Scheduling(machines_t *machines, lots_t *lots);

#endif
