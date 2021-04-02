#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <include/job_base.h>
#include <vector>
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#include "test_job_base.h"

#define amount 100000

using namespace std;


class TestJobBase : public testing::Test{
protected:
	// Job * jb;
	Job ** jb_host;
	Job ** device_jb_addresses;
	Job ** jb_device;
	unsigned int * result_device;
	unsigned int * result_host;
	double arrayOfMsGene[amount];
	double * array_of_ms_gene_device;
	unsigned int arrayOfSizePt[amount], arrayOfMcNum[amount];
	void SetUp() override;
	void TearDown() override;
	void copyArrayOfJobBase(Job **, Job **);
	void setMsGeneData();
};

void TestJobBase::SetUp() {
	// initialize jb_host_*
	size_t sizeof_array_of_pointer = sizeof(Job*) * amount;
	size_t sizeof_array_of_result = sizeof(unsigned int) * amount;

	// host memory allocation
	jb_host = (Job **)malloc(sizeof_array_of_pointer);
	device_jb_addresses = (Job **)malloc(sizeof_array_of_pointer);
	result_host = (unsigned int *)malloc(sizeof_array_of_result);

	// device memory allocation
	cudaMalloc((void **)&jb_device, sizeof_array_of_pointer);
	cudaMalloc((void **)&result_device, sizeof_array_of_result);

	setMsGeneData();
	// initializae host array
	for(unsigned int i = 0 ;i < amount; ++i){
		jb_host[i] = newJob(arrayOfSizePt[i]);	
		// jb_host[i] = new JobBaseChild(i);
		// jb_host[i]->setMsGenePointer(&arrayOfMsGene[i]);
		// jb_host[i]->setProcessTime(NULL, arrayOfSizePt[i]);
	}
	//initilize device array
	copyArrayOfJobBase(device_jb_addresses, jb_host);

	// copy content from host to device
	ASSERT_EQ(cudaMemcpy(jb_device, device_jb_addresses, sizeof_array_of_pointer, cudaMemcpyHostToDevice), CUDA_SUCCESS);
}

void TestJobBase::copyArrayOfJobBase(Job** device_address, Job** src){
	Job * device_temp_jb;
	size_t size = sizeof(Job);
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&device_temp_jb, size), cudaSuccess);
		ASSERT_EQ(cudaMemcpy(device_temp_jb, src[i], size, cudaMemcpyHostToDevice), cudaSuccess);
		device_address[i] = device_temp_jb;
	}
}


void TestJobBase::TearDown(){
	// delete result_host;

	// for(unsigned int i = 0; i < amount; ++i){
	// 	// free host object
	// 	delete jb_host[i];
	// }
	// 
	// // free array
	// delete [] jb_host;
	// delete [] device_jb_addresses;
	// // free device array
	// ASSERT_EQ(cudaFree(jb_device), CUDA_SUCCESS);
	// ASSERT_EQ(cudaFree(result_device), CUDA_SUCCESS);
}

void TestJobBase::setMsGeneData(){
    ifstream file;
    file.open("./ms_data.txt", ios::in);
    if (file){
		for(unsigned int i = 0; i < amount; ++i){
			file >> arrayOfMsGene[i] >> arrayOfSizePt[i] >> arrayOfMcNum[i];
		}
        
    }else {
        cout << "Unable to open file\n";
    }
    file.close();
}

__global__ void testMachineSelection(Job ** jb, unsigned int * result, double * msgene_device, unsigned int numElements){
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numElements){
		jb[id]->base.init = initJobBase;
		jb[id]->base.init(&jb[id]->base);
		jb[id]->base.setMsGenePointer(&jb[id]->base, &(msgene_device[id]));
		result[id] = jb[id]->base.machineSelection(&jb[id]->base);
	}
}


TEST_F(TestJobBase, test_machine_selection_host){
	for(int i = 0; i < amount; ++i){
		jb_host[i]->base.setMsGenePointer(&jb_host[i]->base, &arrayOfMsGene[i]);
		ASSERT_EQ(jb_host[i]->base.machineSelection(&jb_host[i]->base), arrayOfMcNum[i]) << "Entry : "<<i<<std::endl;
	}	
}



TEST_F(TestJobBase, test_machine_selection_device){
	// copy the array content from host to device
	double * msgene_device;
	size_t size_arr = sizeof(double) * amount;
	ASSERT_EQ(cudaMalloc((void**)&msgene_device, size_arr), CUDA_SUCCESS);
	ASSERT_EQ(cudaMemcpy(msgene_device, arrayOfMsGene, size_arr, cudaMemcpyHostToDevice), CUDA_SUCCESS);
	// computing
	testMachineSelection<<<256, 1024>>>(jb_device, result_device, msgene_device, amount);
	// copy the array content from device to host
	size_t size = sizeof(unsigned int) * amount;
	ASSERT_EQ(cudaMemcpy(result_host, result_device, size, cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	
	// testing
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(result_host[i], arrayOfMcNum[i]) << "Entry : " << i << std::endl;
	}
}

