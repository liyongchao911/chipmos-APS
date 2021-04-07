#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>
#include <include/linked_list.h>
#include <include/machine_base.h>
#include <include/job_base.h>
#include <include/common.h>

#include "test_machine_base.h"

#define amount 5000

class TestMachineBase : public testing::Test{
public:
	int *values[amount];
	int **result_deivce;
	int **result_device_arr;
	// Machine ** machines;
	Machine ** machines_device_addresses;
	Machine ** machines_device;

	// Job *** jobs;
	Job *** jobs_device;
	Job *** job_device_addresses;
	Job *** job_device_array_addresses;

	unsigned int sizes[amount];
	unsigned int *sizes_device;
	void SetUp();
	// void TearDown();
};

void TestMachineBase::SetUp(){
	Job * job_device_tmp;
	Job * job_tmp;
	Job ** job_device_arr;
	job_device_addresses = (Job ***)malloc(sizeof(Job**)*amount);
	job_device_array_addresses = (Job***)malloc(sizeof(Job**)*amount);


	for(int i = 0; i < amount; ++i){
		sizes[i] = rand() % 100 + 1; // 

		job_device_addresses[i] = (Job **)malloc(sizeof(Job	*)*sizes[i]);
		values[i] = (int*)malloc(sizeof(int)*sizes[i]);
		for(unsigned int j = 0; j < sizes[i]; ++j){
			values[i][j] = rand() % 100;
			job_tmp = newJob(values[i][j]);
			cudaMalloc((void**)&job_device_tmp, sizeof(Job));
			cudaMemcpy(job_device_tmp, job_tmp, sizeof(Job), cudaMemcpyHostToDevice);
			job_device_addresses[i][j] = job_device_tmp;	
		}
		cudaMalloc((void**)&job_device_arr, sizeof(Job*)*sizes[i]);
		cudaMemcpy(job_device_arr, job_device_addresses[i], sizeof(Job*)*sizes[i], cudaMemcpyHostToDevice);
		job_device_array_addresses[i] = job_device_arr;
	}

	cudaMalloc((void**)&jobs_device, sizeof(Job**)*amount);
	cudaMemcpy(jobs_device, job_device_array_addresses, sizeof(Job**)*amount, cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&sizes_device, sizeof(int)*amount);
	cudaMemcpy(sizes_device, sizes, sizeof(int)*amount, cudaMemcpyHostToDevice);
	

	//***********************************************************************************//
	
	machines_device_addresses = (Machine**)malloc(sizeof(Machine*)*amount);
	Machine *machine_tmp;
	for(int i = 0; i < amount; ++i){
		cudaMalloc((void**)&machine_tmp, sizeof(Machine));	
		machines_device_addresses[i] = machine_tmp;
	}
	cudaMalloc((void**)&machines_device, sizeof(Machine*)*amount);
	cudaMemcpy(machines_device, machines_device_addresses, sizeof(Machine*)*amount, cudaMemcpyHostToDevice);

	result_device_arr = (int**)malloc(sizeof(int*)*amount);
	int *result_tmp;
	for(int i = 0; i < amount; ++i){
		cudaMalloc((void**)&result_tmp, sizeof(int)*sizes[i]);
		result_device_arr[i] = result_tmp;
	}
	cudaMalloc((void**)&result_deivce, sizeof(int*)*amount);
	cudaMemcpy(result_deivce, result_device_arr, sizeof(int*)*amount, cudaMemcpyHostToDevice);
}

__global__ void initMachinesKernel(Machine ** machines, int am){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < am){
		machines[idx]->base.init = initMachineBase;
		// machines[idx]->base.init(&machines[idx]->base);
		initMachine(machines[idx]);
	}
}

__global__ void initJobsKernel(Job *** jobs, unsigned int *sizes, int am){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < am){
		for(int i = 0; i < sizes[idx];++i){
			initJob(jobs[idx][i]);
		}
	}
}

__global__ void addJobsKernel(Machine ** machines, Job *** jobs, unsigned int *sizes, int **result, int am){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < am){
		for(int i = 0; i < sizes[idx]; ++i){
			machines[idx]->base.addJob(&machines[idx]->base, jobs[idx][i]);		
			// __addJob(&machines[idx]->base, &jobs[idx][i]->ele);
		}
		LinkedListElement *ele;
		ele = machines[idx]->base.root;
		for(unsigned int i = 0; i < sizes[idx] ; ++i){
			result[idx][i] = ele->getValue(ele);
			ele = ele->next;
		}
	}
}

TEST_F(TestMachineBase, test_machine_base_add_job){
	initJobsKernel<<<20, 256>>>(jobs_device, sizes_device, amount);	
	initMachinesKernel<<<20, 256>>>(machines_device, amount);
	addJobsKernel<<<20, 256>>>(machines_device, jobs_device, sizes_device, result_deivce,  amount);

	int *result_tmp;
	int ** arr = (int**)malloc(sizeof(int*)*amount);
	ASSERT_EQ(cudaMemcpy(arr, result_deivce, sizeof(int*)*amount, cudaMemcpyDeviceToHost), cudaSuccess);
	for(int i = 0; i < amount; ++i){
		result_tmp = (int*)malloc(sizeof(int)*sizes[i]);
		ASSERT_EQ(cudaMemcpy(result_tmp, arr[i], sizeof(int)*sizes[i], cudaMemcpyDeviceToHost), cudaSuccess);
		for(int j = 0; j < sizes[i]; ++j){
			ASSERT_EQ(result_tmp[j], values[i][j]);
		}
		free(result_tmp);
	}
}

__global__ void sortJobsKernel(Machine ** machines, Job *** jobs, unsigned int *sizes, int **result, int am){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < am){
		for(int i = 0; i < sizes[idx]; ++i){
			__addJob(&machines[idx]->base, &jobs[idx][i]->ele);
		}
		

		machines[idx]->base.sortJob(&machines[idx]->base);

		LinkedListElement *ele;
		ele = machines[idx]->base.root;
		for(unsigned int i = 0; i < sizes[idx] ; ++i){
			result[idx][i] = jobGetValue(ele);
			ele = ele->next;
		}
		
	}
}

TEST_F(TestMachineBase, test_machine_base_sort_job){
	initJobsKernel<<<20, 256>>>(jobs_device, sizes_device, amount);	
	initMachinesKernel<<<20, 256>>>(machines_device, amount);
	sortJobsKernel<<<20, 256>>>(machines_device, jobs_device, sizes_device, result_deivce,  amount);
	
	int *result_tmp;
	int ** arr = (int**)malloc(sizeof(int*)*amount);
	ASSERT_EQ(cudaMemcpy(arr, result_deivce, sizeof(int*)*amount, cudaMemcpyDeviceToHost), cudaSuccess);
	for(int i = 0; i < amount; ++i){
		result_tmp = (int*)malloc(sizeof(int)*sizes[i]);
		qsort(values[i], sizes[i], sizeof(int), cmpint);
		ASSERT_EQ(cudaMemcpy(result_tmp, arr[i], sizeof(int)*sizes[i], cudaMemcpyDeviceToHost), cudaSuccess);
		for(int j = 0; j < sizes[i]; ++j){
			ASSERT_EQ(result_tmp[j], values[i][j]);
		}
		free(result_tmp);
	}

}
