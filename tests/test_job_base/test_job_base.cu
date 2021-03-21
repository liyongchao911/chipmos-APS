#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <include/job_base.h>
#include <vector>
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#define amount 5000

using namespace std;

class JobBaseChild : public JobBase{
private:
	double value;
public:
	JobBaseChild() : JobBase(){
		value = 0;
	}
	JobBaseChild(double val):JobBase(){
		value = val;	
	}

	virtual __device__ __host__ double getValue(){
		return value;
	}
};


class TestJobBase : public testing::Test{
protected:
	JobBase * jb;
	JobBase ** jb_host;
	JobBase ** device_jb_addresses;
	JobBase ** jb_device;
	unsigned int * result_device;
	unsigned int * result_host;
	double arrayOfMsGene[5000];
	unsigned int arrayOfSizePt[5000], arrayOfMcNum[5000];
	void SetUp() override;
	void TearDown() override;
	void copyArrayOfJobBase(JobBase **, JobBase **);
	void setMsGeneData();
};

void TestJobBase::SetUp() {
	// initialize jb_host_*
	size_t sizeof_array_of_pointer = sizeof(JobBase*) * amount;
	size_t sizeof_array_of_result = sizeof(unsigned int) * amount;

	// host memory allocation
	jb_host = (JobBase **)malloc(sizeof_array_of_pointer);
	device_jb_addresses = (JobBase **)malloc(sizeof_array_of_pointer);
	result_host = (unsigned int *)malloc(sizeof_array_of_result);

	// device memory allocation
	cudaMalloc((void **)&jb_device, sizeof_array_of_pointer);
	cudaMalloc((void **)&result_device, sizeof_array_of_result);

	setMsGeneData();
	// initializae host array
	for(unsigned int i = 0 ;i < amount; ++i){
		jb_host[i] = new JobBaseChild(i);
		jb_host[i]->setMsGenePointer(&arrayOfMsGene[i]);
		jb_host[i]->setProcessTime(NULL, arrayOfSizePt[i]);
	}
	//initilize device array
	copyArrayOfJobBase(device_jb_addresses, jb_host);

	// copy content from host to device
	ASSERT_EQ(cudaMemcpy(jb_device, device_jb_addresses, sizeof_array_of_pointer, cudaMemcpyHostToDevice), CUDA_SUCCESS);
}

void TestJobBase::copyArrayOfJobBase(JobBase ** device_address, JobBase ** src){
	JobBase * device_temp_jb;
	size_t size = sizeof(JobBase);
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&device_temp_jb, size), CUDA_SUCCESS);
		ASSERT_EQ(cudaMemcpy(device_temp_jb, src[i], size, cudaMemcpyHostToDevice), CUDA_SUCCESS);
		device_address[i] = device_temp_jb;
	}
}


void TestJobBase::TearDown(){
	delete result_host;

	for(unsigned int i = 0; i < amount; ++i){
		// free host object
		delete jb_host[i];
	}
	
	// free array
	delete [] jb_host;
	delete [] device_jb_addresses;
	// free device array
	ASSERT_EQ(cudaFree(jb_device), CUDA_SUCCESS);
	ASSERT_EQ(cudaFree(result_device), CUDA_SUCCESS);
}

void TestJobBase::setMsGeneData(){
    ifstream file;
    file.open("/home/shani/Parallel-Genetic-Algorithm/tests/test_job_base/ms_data.txt", ios::in);
    if (file){
		for(unsigned int i = 0; i < 5000; ++i){
			file >> arrayOfMsGene[i] >> arrayOfSizePt[i] >> arrayOfMcNum[i];
		}
        
    }else {
        cout << "Unable to open file\n";
    }
    file.close();
}

__global__ void testMachineSelection(JobBase ** jb, unsigned int * result, double * msgene_device, unsigned int numElements){
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numElements){
		jb[id]->setMsGenePointer(&(msgene_device[id]));
		result[id] = jb[id]->machineSelection();
	}
}

TEST_F(TestJobBase, test_machine_selection_device){
	// copy the array content from host to device
	double * msgene_device;
	size_t size_arr = sizeof(double) * amount;
	ASSERT_EQ(cudaMalloc((void**)&msgene_device, size_arr), CUDA_SUCCESS);
	ASSERT_EQ(cudaMemcpy(msgene_device, arrayOfMsGene, size_arr, cudaMemcpyHostToDevice), CUDA_SUCCESS);
	// computing
	testMachineSelection<<<20, 256>>>(jb_device, result_device, msgene_device, amount);
	// copy the array content from device to host
	size_t size = sizeof(unsigned int) * amount;
	ASSERT_EQ(cudaMemcpy(result_host, result_device, size, cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	
	// testing
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(result_host[i], arrayOfMcNum[i]);
	}

}

