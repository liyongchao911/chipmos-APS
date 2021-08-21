#include <gtest/gtest.h>

#include <string>

#define private public
#define protected public

#include "include/lot.h"
#include "include/lots.h"
#include "include/machine.h"
#include "include/machines.h"

#undef private
#undef protected

using namespace std;

class test_stage2_t
{
private:
public:
};
