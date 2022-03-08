#include <gtest/gtest.h>
#include <cstdlib>

int main(int argc, char **argv)
{
    // initialize the time zone for testing
    char env[20];
    strcpy(env, "TZ=UTC-8");
    putenv(env);
    tzset();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
