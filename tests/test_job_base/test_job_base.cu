#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <include/job_base.h>
#include <vector>
#include <gtest/gtest.h>
#define amount 5000


class TestJobBase : public testing::Test{
protected:
	JobBase * jb;
	JobBase ** jb_host;
	JobBase ** device_jb_addresses;
	JobBase ** jb_device;
	void SetUp() override;
	void TearDown() override;
	void copyArrayOfJobBase(JobBase **, JobBase **);
};

void TestJobBase::SetUp() {
	jb = new JobBase();
	// initialize jb_host_*
	size_t sizeof_job_base = sizeof(JobBase);
	size_t sizeof_array_of_pointer = sizeof(JobBase*) * amount;

	// host memory allocation
	jb_host = (JobBase **)malloc(sizeof_array_of_pointer);
	device_jb_addresses = (JobBase **)malloc(sizeof_array_of_pointer);

	// device memory allocation
	cudaMalloc((void **)&jb_device, sizeof_array_of_pointer);

	// initializae host array
	for(unsigned int i = 0 ;i < amount; ++i){
		jb_host[i] = new JobBase(i);
	}

	//initilizae device array
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
	delete jb;

	for(unsigned int i = 0; i < amount; ++i){
		// free host object
		delete jb_host[i];

		// free device object
		ASSERT_EQ(cudaFree(device_job_addresses[i]), CUDA_SUCCESS);
	}
	
	// free array
	delete [] jb_host;
	
	// free device array
	ASSERT_EQ(cudaFree(jb_device), CUDA_SUCCESS);

}

__global__ void testMachineSelection(JobBase ** jb, unsigned int numElements){
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numElements){
		jb[id]->machineSelection();
	}
}

TEST_F(TestJobBase, test_machine_selection_device){

	// computing
	testMachineSelection<<<20, 256>>>(jb_device, amount);

	// copy the array content from device to host
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMemcpy(jb_host[i], device_jb_addresses[i], sizeof(JobBase), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	}

	// testing
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(jb_host[i]->machineSelection(), 1);
	}

}