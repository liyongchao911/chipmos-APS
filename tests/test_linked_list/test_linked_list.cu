#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <include/linked_list.h>
#include <vector>
#include <gtest/gtest.h>
#define amount 5000

class LinkedListChild : public LinkedList{
private:
	double value;
public:
	LinkedListChild() : LinkedList(){
		value = 0;
	}
	LinkedListChild(double val):LinkedList(){
		value = val;	
	}

	virtual __device__ __host__ double getValue(){
		return value;
	}
};

class TestLinkedList : public testing::Test{
protected:
	LinkedList * ls_current;
	LinkedList * ls_next;
	LinkedList * ls_prev;

	LinkedList ** ls_host_current;
	LinkedList ** ls_host_next;
	LinkedList ** ls_host_prev;
	
	LinkedList ** device_current_addresses;
	LinkedList ** device_next_addresses;
	LinkedList ** device_prev_addresses;

	LinkedList ** ls_device_current;
	LinkedList ** ls_device_next;
	LinkedList ** ls_device_prev;
	void SetUp() override;
	void TearDown() override;
	void copyArrayOfLinkedList(LinkedList **, LinkedList **);
};

void TestLinkedList::SetUp() {

	ls_current = new LinkedListChild();	
	ls_next = new LinkedListChild();
	ls_prev = new LinkedListChild();
	
	// initialize ls_host_*
	size_t sizeof_linked_list = sizeof(LinkedList);
	// size_t sizeof_linked_list_p = sizeof(LinkedList *);
	size_t sizeof_array_of_pointer = sizeof(LinkedList*) * amount;

	// host memory allocation
	ls_host_current = (LinkedList **)malloc(sizeof_array_of_pointer);
	ls_host_next = (LinkedList **)malloc(sizeof_array_of_pointer);
	ls_host_prev = (LinkedList **)malloc(sizeof_array_of_pointer);

	device_next_addresses = (LinkedList **)malloc(sizeof_array_of_pointer);
	device_current_addresses = (LinkedList **)malloc(sizeof_array_of_pointer);
	device_prev_addresses = (LinkedList **)malloc(sizeof_array_of_pointer);

	// device memory allocation
	cudaMalloc((void **)&ls_device_current, sizeof_array_of_pointer);
	cudaMalloc((void**)&ls_device_next, sizeof_array_of_pointer);
	cudaMalloc((void**)&ls_device_prev, sizeof_array_of_pointer);

	// initializae host array
	for(unsigned int i = 0 ;i < amount; ++i){
		ls_host_current[i] = new LinkedListChild(i);
		ls_host_next[i] = new LinkedListChild(i*2);
		ls_host_prev[i] = new LinkedListChild(i*4);
	}

	//initilizae device array
	copyArrayOfLinkedList(device_current_addresses, ls_host_current);
	copyArrayOfLinkedList(device_next_addresses, ls_host_next);
	copyArrayOfLinkedList(device_prev_addresses, ls_host_prev);
	// LinkedList * device_temp_ls;
	// for(unsigned int i = 0; i < amount; ++i){
	// 	ASSERT_EQ(cudaMalloc((void**)&device_temp_ls, sizeof_linked_list), CUDA_SUCCESS);
	// 	ASSERT_EQ(cudaMemcpy(device_temp_ls, ls_host_current[i], sizeof_linked_list, cudaMemcpyHostToDevice), CUDA_SUCCESS);
	// 	device_current_addresses[i] = device_temp_ls;
	// }

	// copy content from host to device
	ASSERT_EQ(cudaMemcpy(ls_device_current, device_current_addresses, sizeof_array_of_pointer, cudaMemcpyHostToDevice), CUDA_SUCCESS);
	ASSERT_EQ(cudaMemcpy(ls_device_next, device_next_addresses, sizeof_array_of_pointer, cudaMemcpyHostToDevice), CUDA_SUCCESS);
	ASSERT_EQ(cudaMemcpy(ls_device_prev, device_prev_addresses, sizeof_array_of_pointer, cudaMemcpyHostToDevice), CUDA_SUCCESS);

}

void TestLinkedList::copyArrayOfLinkedList(LinkedList ** device_address, LinkedList ** src){
	LinkedList * device_temp_ls;
	size_t size = sizeof(LinkedList);
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&device_temp_ls, size), CUDA_SUCCESS);
		ASSERT_EQ(cudaMemcpy(device_temp_ls, src[i], size, cudaMemcpyHostToDevice), CUDA_SUCCESS);
		device_address[i] = device_temp_ls;
	}
}


void TestLinkedList::TearDown(){
	delete ls_current;
	delete ls_next;
	delete ls_prev;

	for(unsigned int i = 0; i < amount; ++i){
		// free host object
		delete ls_host_current[i];
		delete ls_host_next[i];
		delete ls_host_prev[i];

		// free device object
		ASSERT_EQ(cudaFree(device_current_addresses[i]), CUDA_SUCCESS);
		ASSERT_EQ(cudaFree(device_next_addresses[i]), CUDA_SUCCESS);
		ASSERT_EQ(cudaFree(device_prev_addresses[i]), CUDA_SUCCESS);
	}
	
	// free array
	delete [] ls_host_current;
	delete [] ls_host_next;
	delete [] ls_host_prev;
	
	// free device array
	ASSERT_EQ(cudaFree(ls_device_current), CUDA_SUCCESS);
	ASSERT_EQ(cudaFree(ls_device_next), CUDA_SUCCESS);
	ASSERT_EQ(cudaFree(ls_device_prev), CUDA_SUCCESS);

}

TEST_F(TestLinkedList, test_set_next_host){
	ls_current->setNext(ls_next);
	EXPECT_EQ(ls_current->getNext(), ls_next);
}

TEST_F(TestLinkedList, test_set_prev_host){
	ls_current->setPrev(ls_prev);
	EXPECT_EQ(ls_current->getPrev(), ls_prev);
}


__global__ void testSetNext(LinkedList ** current, LinkedList ** next, unsigned int numElements){
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numElements){
		current[id]->setNext(next[id]);
	}
}

TEST_F(TestLinkedList, test_set_next_device){

	// computing
	testSetNext<<<20, 256>>>(ls_device_current,ls_device_next, amount);

	// copy the array content from device to host
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMemcpy(ls_host_current[i], device_current_addresses[i], sizeof(LinkedList), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	}

	// testing
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(ls_host_current[i]->getNext(), device_next_addresses[i]);
	}

}

__global__ void testSetPrev(LinkedList ** current, LinkedList ** prev, unsigned int numElements){
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numElements){
		current[id]->setPrev(prev[id]);
	}
}

TEST_F(TestLinkedList, test_set_prev_device){
	// computing
	testSetPrev<<<20, 256>>>(ls_device_current, ls_device_prev, amount);

	// copy the array content from device to host
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMemcpy(ls_host_current[i], device_current_addresses[i], sizeof(LinkedList), cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	}

	// testing
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(ls_host_current[i]->getPrev(), device_prev_addresses[i])<<"At index : "<< i << std::endl;
	}

}

