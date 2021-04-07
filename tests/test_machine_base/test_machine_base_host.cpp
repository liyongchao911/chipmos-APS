#include <cuda.h>
#include <include/machine_base.h>
#include <gtest/gtest.h>

#include "include/linked_list.h"
#include "test_machine_base.h"

#define amount 5000


class TestMachineBaseHost : public testing::Test{
public:
	int ** values;
	int sizes[amount];
	Job ***jobs;
	Machine **machines;
	void SetUp()override;
};

void TestMachineBaseHost::SetUp(){
	jobs = (Job***)malloc(sizeof(Job**)*amount);
	values = (int**)malloc(sizeof(int*)*amount);
	machines = (Machine**)malloc(sizeof(Machine*)*amount);

	for(int i = 0; i < amount; ++i){
		sizes[i] = rand() % 1000 + 1;
		values[i] = (int*)malloc(sizeof(int)*sizes[i]);
		jobs[i] = (Job**)malloc(sizeof(Job*)*sizes[i]);
		machines[i] = newMachine();
		for(int j = 0; j < sizes[i]; ++j){
			values[i][j] = rand() % 1024;
			jobs[i][j] = newJob(values[i][j]);
		}	
	}

}

TEST_F(TestMachineBaseHost, test_machine_base_host_add_job){
	for(int i = 0; i < amount; ++i){
		for(int j = 0; j < sizes[i]; ++j){
			machines[i]->base.addJob(&machines[i]->base, jobs[i][j]);
		}
	}
	
	LinkedListElement *ele;
	for(int i = 0; i < amount; ++i){
		ele = machines[i]->base.root;
		for(int j = 0; j < sizes[i]; ++j){
			ASSERT_EQ(ele->getValue(ele), values[i][j]) << "i = "<< i <<"j = "<<j<<std::endl;
			ele = ele->next;
		}	
	}
}

TEST_F(TestMachineBaseHost, test_machine_base_host_sort_job){
	for(int i = 0; i < amount; ++i){
		for(int j = 0; j < sizes[i]; ++j){
			machines[i]->base.addJob(&machines[i]->base, jobs[i][j]);
		}
	}

	for(int i = 0; i < amount; ++i){
		qsort(values[i], sizes[i], sizeof(int), cmpint);
		machines[i]->base.sortJob(&machines[i]->base);
	}

	LinkedListElement *ele;
	for(int i = 0; i < amount; ++i){
		ele = machines[i]->base.root;
		for(int j = 0; j < sizes[i]; ++j){
			ASSERT_EQ(ele->getValue(ele), values[i][j]) << "i = "<< i <<"j = "<<j<<std::endl;
			ele = ele->next;
		}	
	}
}
