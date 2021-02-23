#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <include/linked_list.h>
#include <gtest/gtest.h>

class TestLinkedList : public testing::Test{
protected:
	LinkedList * ls;
	TestLinkedList();
	LinkedList * testSetNextHost(LinkedList * next);
	LinkedList * testSetLastHost(LinkedList * last);
	LinkedList * testGetNextHost();
	LinkedList * testGetLastHost();
};

TestLinkedList::TestLinkedList(){
	ls = new LinkedList();	
}

LinkedList * TestLinkedList::testGetLastHost(){
	return ls->getLast();
}

LinkedList * TestLinkedList::testGetNextHost(){
	return ls->getNext();
}

LinkedList * TestLinkedList::testSetNextHost(LinkedList * next){
	ls->setNext(next);
	return ls->next;
}

LinkedList * TestLinkedList::testSetLastHost(LinkedList * last){
	ls->setLast(last);
	return ls->last;
}



TEST_F(TestLinkedList, test_set_next_host){
	LinkedList * test1 = new LinkedList();
	EXPECT_EQ(testSetNextHost(test1), test1);
	delete test1;
}

TEST_F(TestLinkedList, test_set_last_host){
	LinkedList * test2 = new LinkedList();
	EXPECT_EQ(testSetLastHost(test2), test2);
	delete test2;
}

TEST_F(TestLinkedList, test_get_next_host){
	LinkedList * test = new LinkedList();
	ls->setNext(test);
	EXPECT_EQ(this->testGetNextHost(), test);
	delete test;
}
TEST_F(TestLinkedList, test_get_last_host){
	LinkedList * test = new LinkedList();
	ls->setLast(test);
	EXPECT_EQ(this->testGetLastHost(), test);
	delete test;
}

__global__ void testSetNext(LinkedList ** current, LinkedList ** next, unsigned int numElements){
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numElements){
		current[id]->setNext(next[id]);
	}
}

TEST_F(TestLinkedList, test_set_next_device){
	unsigned int amount = 5000;

	LinkedList ** host_current;
	LinkedList ** host_next;
	LinkedList ** device_current;
	LinkedList ** device_next;
	LinkedList ** device_current_addresses;
	LinkedList ** device_next_addresses;

	size_t sizeof_linked_list = sizeof(LinkedList);
	size_t sizeof_linked_list_p = sizeof(LinkedList *);
	
	// host memory allocation
	host_current = (LinkedList **)malloc(sizeof_linked_list_p * amount);	
	host_next = (LinkedList **)malloc(sizeof_linked_list_p * amount);
	device_current_addresses = (LinkedList **)malloc(sizeof_linked_list_p * amount);
	device_next_addresses = (LinkedList **)malloc(sizeof_linked_list_p * amount);
	// device memory allocation
	cudaMalloc((void **)&device_current, sizeof_linked_list_p * amount);
	cudaMalloc((void **)&device_next, sizeof_linked_list_p * amount);
	
	// initialize host array
	for(unsigned int i = 0; i < amount; ++i){
		host_current[i] = new LinkedList(i);
		host_next[i] = new LinkedList(i * 2);	
	}


	// initialize device array
	LinkedList * device_temp_ls;
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&device_temp_ls, sizeof_linked_list), CUDA_SUCCESS);
		ASSERT_EQ(cudaMemcpy(device_temp_ls, host_current[i], sizeof_linked_list, cudaMemcpyHostToDevice), CUDA_SUCCESS);
		device_current_addresses[i] = device_temp_ls;
	}
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&device_temp_ls, sizeof_linked_list), CUDA_SUCCESS);
		ASSERT_EQ(cudaMemcpy(device_temp_ls, host_next[i], sizeof_linked_list, cudaMemcpyHostToDevice), CUDA_SUCCESS);
		device_next_addresses[i] = device_temp_ls;
	}
	ASSERT_EQ(cudaMemcpy(device_current, device_current_addresses, sizeof_linked_list_p * amount, cudaMemcpyHostToDevice), CUDA_SUCCESS);
	ASSERT_EQ(cudaMemcpy(device_next, device_next_addresses, sizeof_linked_list_p * amount, cudaMemcpyHostToDevice), CUDA_SUCCESS);


	// computing
	testSetNext<<<20, 256>>>(device_current, device_next, amount);


	// copy the array content from device to host
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMemcpy(host_current[i], device_current_addresses[i], sizeof_linked_list, cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	}
		// testing
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(host_current[i]->getNext(), device_next_addresses[i]);
	}

	// free host memory
	for(unsigned int i = 0; i < amount; ++i){
		delete host_current[i];
		delete host_next[i];
	}
	delete[] host_current;
	delete[] host_next;

	// free device memory
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaFree(device_current_addresses[i]), CUDA_SUCCESS);
		ASSERT_EQ(cudaFree(device_next_addresses[i]), CUDA_SUCCESS);
	}
	delete[] device_next_addresses;
	delete[] device_current_addresses;

	ASSERT_EQ(cudaFree(device_current), CUDA_SUCCESS);
	ASSERT_EQ(cudaFree(device_next), CUDA_SUCCESS);
	
}

__global__ void testSetLast(LinkedList ** current, LinkedList ** next, unsigned int numElements){
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numElements){
		current[id]->setLast(next[id]);
	}
}

TEST_F(TestLinkedList, test_set_last_device){
	unsigned int amount = 5000;

	LinkedList ** host_current;
	LinkedList ** host_last;
	LinkedList ** device_current;
	LinkedList ** device_last;
	LinkedList ** device_current_addresses;
	LinkedList ** device_last_addresses;

	size_t sizeof_linked_list = sizeof(LinkedList);
	size_t sizeof_linked_list_p = sizeof(LinkedList *);
	
	// host memory allocation
	host_current = (LinkedList **)malloc(sizeof_linked_list_p * amount);	
	host_last = (LinkedList **)malloc(sizeof_linked_list_p * amount);
	device_current_addresses = (LinkedList **)malloc(sizeof_linked_list_p * amount);
	device_last_addresses = (LinkedList **)malloc(sizeof_linked_list_p * amount);
	// device memory allocation
	cudaMalloc((void **)&device_current, sizeof_linked_list_p * amount);
	cudaMalloc((void **)&device_last, sizeof_linked_list_p * amount);
	
	// initialize host array
	for(unsigned int i = 0; i < amount; ++i){
		host_current[i] = new LinkedList(i);
		host_last[i] = new LinkedList(i * 2);	
	}


	// initialize device array
	// current
	LinkedList * device_temp_ls;
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&device_temp_ls, sizeof_linked_list), CUDA_SUCCESS);
		ASSERT_EQ(cudaMemcpy(device_temp_ls, host_current[i], sizeof_linked_list, cudaMemcpyHostToDevice), CUDA_SUCCESS);
		device_current_addresses[i] = device_temp_ls;
	}
	// last
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&device_temp_ls, sizeof_linked_list), CUDA_SUCCESS);
		ASSERT_EQ(cudaMemcpy(device_temp_ls, host_last[i], sizeof_linked_list, cudaMemcpyHostToDevice), CUDA_SUCCESS);
		device_last_addresses[i] = device_temp_ls;
	}
	ASSERT_EQ(cudaMemcpy(device_current, device_current_addresses, sizeof_linked_list_p * amount, cudaMemcpyHostToDevice), CUDA_SUCCESS);
	ASSERT_EQ(cudaMemcpy(device_last, device_last_addresses, sizeof_linked_list_p * amount, cudaMemcpyHostToDevice), CUDA_SUCCESS);


	// computing
	testSetNext<<<20, 256>>>(device_current, device_last, amount);


	// copy the array content from device to host
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMemcpy(host_current[i], device_current_addresses[i], sizeof_linked_list, cudaMemcpyDeviceToHost), CUDA_SUCCESS);
	}
		// testing
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(host_current[i]->getNext(), device_last_addresses[i]);
	}

	// free host memory
	for(unsigned int i = 0; i < amount; ++i){
		delete host_current[i];
		delete host_last[i];
	}
	delete[] host_current;
	delete[] host_last;

	// free device memory
	for(unsigned int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaFree(device_current_addresses[i]), CUDA_SUCCESS);
		ASSERT_EQ(cudaFree(device_last_addresses[i]), CUDA_SUCCESS);
	}
	delete[] device_last_addresses;
	delete[] device_current_addresses;

	ASSERT_EQ(cudaFree(device_current), CUDA_SUCCESS);
	ASSERT_EQ(cudaFree(device_last), CUDA_SUCCESS);
	
}
