#include <gtest/gtest.h>
#include <cstdlib>

int main(int argc, char **argv)
{
    // initialize the time zone for testing
    setenv("TZ", "UTC-8", 1);
    tzset();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
