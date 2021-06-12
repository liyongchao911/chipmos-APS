#include <include/machine.h>

bool ares_ptr_comp(ares_t * a1, ares_t * a2){
    return a1->time > a2->time;
}

bool ares_comp(ares_t a1, ares_t a2){
    return a1.time > a2.time;
}
