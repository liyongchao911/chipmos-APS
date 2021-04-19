#include <gtest/gtest.h>

int JOB_AMOUNT = 20000;
int MACHINE_AMOUNT = 1000;
int CHROMOSOME_AMOUNT = 200;
int GENERATIONS = 100;

int main(int argc, char ** argv){
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
