#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

typedef struct __setup_time_parameters_t {
    double TIME_CWN;
    double TIME_CK;
    double TIME_EU;
    double TIME_MC;
    double TIME_SC;
    double TIME_CSC;
    double TIME_USC;
    double TIME_ICSI;
} setup_time_parameters_t;


typedef struct weights_t {
    float WEIGHT_SETUP_TIMES;
    float WEIGHT_TOTAL_COMPLETION_TIME;
    float WEIGHT_MAX_SETUP_TIMES;
    float WEIGHT_CR;
    float WEIGHT_TR;
} weights_t;

typedef struct scheduling_parameters_t {
    double PEAK_PERIOD;
    int MAX_SETUP_TIMES;
    int MINUTE_THRESHOLD;
} scheduling_parameters_t;

#endif
