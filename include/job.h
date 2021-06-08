#ifndef __JOB_H__
#define __JOB_H__

typedef struct __info_t{
    union{
        char text[32];
        unsigned int number[8];
    }data;
    unsigned int text_size : 5;
    unsigned int number_size : 3;
}job_info_t;

typedef struct __job_t{
    job_info_t part_no;
    job_info_t pin_package;
    char urgent_code;
}job_t;


#endif
