#include <algorithm>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <include/linked_list.h>
#include <ostream>
#include <texture_types.h>
#include <vector>
#include <gtest/gtest.h>
#include <tests/test_linked_list/test_linked_list.h>
#include <string>
#define amount 5000

class TestLinkedList : public testing::Test{
protected:
	LinkedList * ls_current;
	LinkedList * ls_next;
	LinkedList * ls_prev;

	LinkedList ** ls_host_current;
	LinkedList ** ls_host_next;
	LinkedList ** ls_host_prev;
	
	LinkedList ** device_next_addresses;
	LinkedList ** device_current_addresses;
	LinkedList ** device_prev_addresses;

	LinkedList ** ls_device_current;
	LinkedList ** ls_device_next;
	LinkedList ** ls_device_prev;

	LinkedList *** device_elements;
	LinkedList *** array_of_pointer_to_device_address;
	LinkedList *** host_elements;



	double ** host_data;
	double ** device_data;
	double ** array_of_device_pointers;
	int row_counts;
	unsigned int host_sizes[amount];
	unsigned int *device_sizes;
	

	void SetUp() override;
	void TearDown() override;
	void basicFunctionTestingSetUp();
	void advanceFunctionTestingSetUp();
	void kernelSortingFunctionTestingSetup();
	void hostSortingFunctionTestingSetup();
	void copyArrayOfLinkedList(LinkedList **, LinkedList **);
	bool varifySortingAlgorithm(LinkedList *head, unsigned int size);
};

std::vector<double> split(char *str, char *delim) {
	std::vector<double> data;
	char *token = NULL;
	char *saveptr = NULL;

	token = strtok_r(str, delim, &saveptr);

	while(token != NULL){
		data.push_back(atof(token));
		token = strtok_r(NULL, delim, &saveptr);
	}
	return data;
}

bool TestLinkedList::varifySortingAlgorithm(LinkedList *head, unsigned int size){
	if(!head)
		return false;
	
	LinkedList *iter = head->getNext();
	LinkedList *prev = head;
	unsigned int count = 1;
	while(iter){
		if(iter->getValue() < prev->getValue()){
			return false;	
		}
		prev = iter;
		iter = iter->getNext();
		++count;	
	}
	
	if(count != size)
		return false;


	return true;
}



void TestLinkedList::advanceFunctionTestingSetUp(){
	// malloc host_data
	host_data = (double**)malloc(sizeof(double*)*amount);

	// setup host data
	FILE * file = NULL;
	file = fopen("./data.txt", "r");
	if(!file){
		perror("Cannot open file\n");
		exit(-1);
	}
	char buf[1000000];
	char delim[2] = {',', '\0'};
	std::vector<double> data;
	// int count = 0;
	while(fscanf(file, "%s", buf) != EOF){
		data = split(buf, delim);
		host_sizes[row_counts] = data.size();
		host_data[row_counts] = (double*)malloc(sizeof(double)*host_sizes[row_counts]);
		if(!host_data[row_counts]){
			perror("advanceFunctionTestingSetUp malloc failed!\n");
			exit(-1);
		}
		for(unsigned int j = 0, size = data.size(); j < size; ++j){
			host_data[row_counts][j] = data[j];
		}
		++row_counts;
	}
	fclose(file);
	printf("Inputing all data(%d) is done\n", row_counts+1);
}

void TestLinkedList::kernelSortingFunctionTestingSetup(){
	array_of_device_pointers = (double**)malloc(sizeof(double*)*amount);
	ASSERT_EQ(cudaMalloc((void**)&device_data, sizeof(double*)*amount), cudaSuccess);

	double *temp;
	for(int i = 0; i < row_counts; ++i){
		ASSERT_EQ(cudaMalloc((void**)&temp, sizeof(double)*host_sizes[i]), cudaSuccess);
		ASSERT_EQ(cudaMemcpy(temp, host_data[i], sizeof(double)*host_sizes[i], cudaMemcpyHostToDevice), cudaSuccess);
		array_of_device_pointers[i] = temp;
	}
	ASSERT_EQ(cudaMemcpy(device_data, array_of_device_pointers, sizeof(double*)*amount, cudaMemcpyHostToDevice), cudaSuccess);
	printf("Finish copy all data from host to device\n");
	ASSERT_EQ(cudaMalloc((void**)&device_sizes, sizeof(unsigned int)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(device_sizes, host_sizes, sizeof(unsigned int)*amount, cudaMemcpyHostToDevice), cudaSuccess);	
	printf("Finish copy size data from host to device\n");

	array_of_pointer_to_device_address = (LinkedList ***)malloc(sizeof(LinkedList **)*amount);
	LinkedList **linked_lists_temp;
	for(int i = 0; i < row_counts;++i){
		ASSERT_EQ(cudaMalloc((void**)&linked_lists_temp, sizeof(LinkedList*)*host_sizes[i]), cudaSuccess);
		array_of_pointer_to_device_address[i] = linked_lists_temp;
	}
	ASSERT_EQ(cudaMalloc((void**)&device_elements, sizeof(LinkedList **)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(device_elements, array_of_pointer_to_device_address, sizeof(LinkedList **)*amount, cudaMemcpyHostToDevice), cudaSuccess);
	printf("Finish create empty 2-D device array");
}

void TestLinkedList::hostSortingFunctionTestingSetup(){
	// for(int i = 0; i < row_counts; ++i){
	// 	for(unsigned int j = 1; j < sizes[i] - 1; ++j){
	// 		ls_elements_host_arrays[i][j]->setNext(ls_elements_host_arrays[i][j + 1]);
	// 		ls_elements_host_arrays[i][j]->setPrev(ls_elements_host_arrays[i][j - 1]);
	// 	}
	// 	if(sizes[i] > 1){
	// 		ls_elements_host_arrays[i][0]->setNext(ls_elements_host_arrays[i][1]);
	// 		ls_elements_host_arrays[i][sizes[i] - 1]->setPrev(ls_elements_host_arrays[i][sizes[i] - 2]);
	// 	}
	// }
}


void TestLinkedList::SetUp() {
	row_counts = 0;
	basicFunctionTestingSetUp();
}

void TestLinkedList::basicFunctionTestingSetUp() {
	ls_current = new LinkedListChild();	
	ls_next = new LinkedListChild();
	ls_prev = new LinkedListChild();
	
	// initialize ls_host_*
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
/**
 * @brief Initialize the liked list before sorting
 * @param elements 2-D array 
 *
 * @detail Connect elements in array and form the linked list
 */
__global__ void sortingSetup(LinkedList *** elements, double ** data, unsigned int * amount_of_data,  unsigned int amounts, cudaError_t *err){
	// for(unsigned int i = 0; i < amounts; ++i){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < amounts){
		err[i] = cudaSuccess;
		LinkedListChild * temp;
		for(unsigned int j = 0; j < amount_of_data[i]; ++j){
			// temp = (LinkedListChild *)malloc(sizeof(LinkedListChild));
			temp = new LinkedListChild(data[i][j]);
			if(!temp){
				err[i] = cudaErrorInvalidDevicePointer;
				return;
			}
			// temp->setValue(data[i][j]);
			elements[i][j] = temp;
			
		}
		__syncthreads();
		if(amount_of_data[i] >= 3){
			for(unsigned int j = 1; j < amount_of_data[i] - 1; ++j){
				elements[i][j]->setNext(elements[i][j + 1]);
				elements[i][j]->setPrev(elements[i][j - 1]);
			}
		}

		if(amount_of_data[i] > 1){
			elements[i][0]->setNext(elements[i][1]); 
			elements[i][ amount_of_data[i] - 1 ]->setPrev(elements[i][ amount_of_data[i] - 2]); 
		}
	 	__syncthreads();
		
		double val;
		LinkedList *iter = elements[i][0];
		for(unsigned int j = 1; j < amount_of_data[i] - 1; ++j){
			if(iter->getNext() != elements[i][j]){
				err[i] = cudaErrorInvalidValue;
				return;
			}
			iter = iter->getNext();
		}
		return;
	}

}

__global__ void sortingLinkedListKernel(LinkedList *** elements, unsigned int * elements_amount_in_an_array,  unsigned int amounts, double ** result){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	LinkedList * iter;
	unsigned int i;
	if(idx < amounts){
		// elements[idx][0]->setNext(NULL);
		elements[idx][0] = linkedListMergeSort(elements[idx][0]);
		iter = elements[idx][0];
		for(i = 0; i < elements_amount_in_an_array[idx]; ++i){
			elements[idx][i] = iter;
			result[idx][i] = iter->getValue();
			iter = iter->getNext();
		}
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

TEST_F(TestLinkedList, test_device_sort_linked_list) {
	this->advanceFunctionTestingSetUp();
	this->kernelSortingFunctionTestingSetup();

	setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
	printf("Finish all setup on host\n");
	cudaError_t *host_errs = (cudaError_t*)malloc(sizeof(cudaError_t)*row_counts);
	cudaError_t *device_errs;
	cudaMalloc((void**)&device_errs, sizeof(cudaError_t)*row_counts);
	sortingSetup<<<20, 256>>>(device_elements, device_data, device_sizes, row_counts, device_errs);
	cudaMemcpy(host_errs, device_errs, sizeof(cudaError_t)*row_counts, cudaMemcpyDeviceToHost);
	for(unsigned int i = 0; i < row_counts; ++i){
		ASSERT_NE(host_errs[i], cudaErrorInvalidDevicePointer);
		ASSERT_NE(host_errs[i], cudaErrorInvalidValue);
		ASSERT_EQ(host_errs[i], cudaSuccess);
	}
	
	printf("Finish Kernel setup\n");
	printf("Start Sorting!!!\n");
	sortingLinkedListKernel<<<20, 256>>>(device_elements, device_sizes, row_counts, device_data);
	printf("Finish sorting\n");	
	double ** result = (double **)malloc(sizeof(double*)*amount);
	ASSERT_EQ(cudaMemcpy(result, device_data, sizeof(double*)*amount, cudaMemcpyDeviceToHost), cudaSuccess);
	double *temp;

	for(unsigned int i = 0; i < row_counts; ++i){
		temp = (double*)malloc(sizeof(double)*host_sizes[i]);
		ASSERT_EQ(cudaMemcpy(temp, result[i], sizeof(double)*host_sizes[i], cudaMemcpyDeviceToHost), cudaSuccess);
		for(unsigned int j = 1; j < host_sizes[i]; ++j){
			ASSERT_LE(temp[j - 1], temp[j]);
		}
		// printf("Pass row %d\n",i);
	}
	// size_t stackSize = 4096;
	// ASSERT_EQ(cudaDeviceSetLimit(cudaLimitStackSize, stackSize), CUDA_SUCCESS);
}

// TEST_F(TestLinkedList, test_host_sort_linked_list) {
// 	this->advanceFunctionTestingSetUp();
// 	this->hostSortingFunctionTestingSetup();
// 	
// 	// varify element is connect with each 
// 	LinkedList * iter;
// 	for(unsigned int i = 0; i < row_counts; ++i){
// 		iter = ls_elements_host_arrays[i][0];
// 		int j = 0;
// 		while(iter){
// 			ASSERT_EQ(iter, ls_elements_host_arrays[i][j]);
// 			iter = iter->getNext();
// 			++j;
// 		}
// 	}
// 	for(unsigned int i = 0; i < row_counts; ++i){
// 		ls_elements_host_arrays[i][0] = linkedListMergeSort(ls_elements_host_arrays[i][0]);
// 		iter = ls_elements_host_arrays[i][0];
// 		for(unsigned int j = 0; j < sizes[i]; ++j){
// 			ls_elements_host_arrays[i][j] = iter;
// 			iter = iter->getNext();
// 		}
// 		ASSERT_EQ(varifySortingAlgorithm(ls_elements_host_arrays[i][0], sizes[i]), true);
// 	}
// 
// }
