#include <gtest/gtest.h>

int JOB_AMOUNT = 50000;
int MACHINE_AMOUNT = 10000;
int CHROMOSOME_AMOUNT = 100;
int GENERATIONS = 100;

int main(int argc, char ** argv){
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
