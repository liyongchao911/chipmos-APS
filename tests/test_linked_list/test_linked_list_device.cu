#include <cstdio>
#include <ctime>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <include/linked_list.h>
#include <gtest/gtest.h>
#include <cuda.h>
#include <regex.h>
#include <texture_types.h>
#include <time.h>
#include <tests/def.h>

#define amount 5000

#include "test_linked_list.h"

class TestLinkedListDevice : public testing::Test{
public:
	int values[amount][amount];	

	LinkedListItem ***item_address_on_device;
	LinkedListItem ***items;



	int sizes[amount];
	void SetUp() override;
	void TearDown() override;

	void advanceSetup();

};

void TestLinkedListDevice::TearDown(){

}



void TestLinkedListDevice::advanceSetup(){
	item_address_on_device = (LinkedListItem ***)malloc(sizeof(LinkedListItem**)*amount);	

	LinkedListItem *item_device;
	int count = 0;
	for(int i = 0; i < amount; ++i){
		sizes[i] = rand() % 200 + 800;
		count += sizes[i];
		item_address_on_device[i] = (LinkedListItem **)malloc(sizeof(LinkedListItem*)*sizes[i]);
		for(int j = 0; j < sizes[i]; ++j){
			values[i][j] = rand() % 1024;
			ASSERT_EQ(cudaMalloc((void**)&item_device, sizeof(LinkedListItem)), cudaSuccess);
			item_address_on_device[i][j] = item_device;
		}
	}
	
	LinkedListItem ** items_array;
	LinkedListItem *** items_array_of_array = (LinkedListItem ***)malloc(sizeof(LinkedListItem **)*amount);
	for(int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&items_array, sizeof(LinkedListItem*)*sizes[i]), cudaSuccess);
		ASSERT_EQ(cudaMemcpy(items_array, item_address_on_device[i], sizeof(LinkedListItem*)*sizes[i], cudaMemcpyHostToDevice), cudaSuccess);
		items_array_of_array[i] = items_array;
	}

	ASSERT_EQ(cudaMalloc((void**)&items, sizeof(LinkedListItem**)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(items, items_array_of_array, sizeof(LinkedListItem**)*amount, cudaMemcpyHostToDevice), cudaSuccess);
	PRINTF("Amount of testing elements is %d\n", count);
}


__global__ void sortingSetUp(LinkedListItem ***items,  int *sizes, int **values){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < amount){
		// first initial all items;
		// connect to device function
		for(int i = 0; i < sizes[idx]; ++i){
			items[idx][i]->ele.getValue = linkedListItemGetValue;
			items[idx][i]->ele.setNext = __listEleSetNext;
			items[idx][i]->ele.setPrev = __listEleSetPrev;
			items[idx][i]->ele.ptr_derived_object = items[idx][i];
			items[idx][i]->value = values[idx][i];
		}

		for(int i = 0, size = sizes[idx] - 1; i < size; ++i){
			items[idx][i]->ele.setNext(&items[idx][i]->ele, &items[idx][i + 1]->ele);
		}

	}
}

__global__ void sorting(LinkedListItem ***items, int **values, int am){
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
 	if(idx < am){
		LinkedListElement *iter;
 		iter = linkedListMergeSort(&(items[idx][0]->ele));
		items[idx][0] = (LinkedListItem*)iter->ptr_derived_object;
		iter = &(items[idx][0]->ele);

		for(int i = 0; iter ; ++i){
			values[idx][i] = iter->getValue(iter);
			iter = iter->next;
		}
		
		// for(int j = 0; j < am; ++j){
		// 	iter = &(items[j][0]->ele);
		// 	for(int i = 0; iter ; ++i){
		// 		values[j][i] = iter->getValue(iter);
		// 		iter = iter->next;
		// 	}
		// }

 	}
}


void TestLinkedListDevice::SetUp(){
	
}


TEST_F(TestLinkedListDevice, test_sort_linked_list_on_device){

	/**********ALLOCAT result array***********/
	advanceSetup();
	int ** result_arr;
	int **result_arr_device;
	int *result_tmp;
	result_arr = (int**)malloc(sizeof(int*)*amount);
	for(int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&result_tmp, sizeof(int)*sizes[i]), cudaSuccess);
		result_arr[i] = result_tmp;
	}
	ASSERT_EQ(cudaMalloc((void**)&result_arr_device, sizeof(int*)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(result_arr_device, result_arr, sizeof(int*)*amount, cudaMemcpyHostToDevice), cudaSuccess);

	int *sizes_device;
	ASSERT_EQ(cudaMalloc((void**)&sizes_device, sizeof(int)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(sizes_device, sizes, sizeof(int)*amount, cudaMemcpyHostToDevice), cudaSuccess);


	/*********ALLOCATE values array and copy value********/
	int **values_arr = (int **)malloc(sizeof(int*)*amount);
	int *values_tmp;
	int **values_arr_device;
	for(int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&values_tmp, sizeof(int)*sizes[i]), cudaSuccess);
		ASSERT_EQ(cudaMemcpy(values_tmp, values[i], sizeof(int)*sizes[i], cudaMemcpyHostToDevice), cudaSuccess);
		values_arr[i] = values_tmp;
	}
	ASSERT_EQ(cudaMalloc((void**)&values_arr_device, sizeof(int*)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(values_arr_device, values_arr, sizeof(int*)*amount, cudaMemcpyHostToDevice), cudaSuccess);
	

	clock_t t1 = clock();
	sortingSetUp<<<200, 256>>>(items, sizes_device, values_arr_device);
	sorting<<<200, 256>>>(items, result_arr_device, amount);
	cudaDeviceSynchronize();
	clock_t t2 = clock();
	PRINTF("Time elapse : %.3fs\n", (t2 - t1) / (double)CLOCKS_PER_SEC);

	ASSERT_EQ(cudaMemcpy(result_arr, result_arr_device, sizeof(int*)*amount, cudaMemcpyDeviceToHost), cudaSuccess);
	for(int i = 0; i < amount; ++i){
		result_tmp = (int*)malloc(sizeof(int)*sizes[i]);
		ASSERT_EQ(cudaMemcpy(result_tmp, result_arr[i], sizeof(int)*sizes[i], cudaMemcpyDeviceToHost), cudaSuccess);
		qsort(values[i], sizes[i], sizeof(int), cmpint);
		// printf("Line %d : ", i);
		for(int j = 0; j < sizes[i]; ++j){
			ASSERT_EQ(values[i][j], result_tmp[j]); 
			// printf("%d ", result_tmp[j]);
		}
		// printf("\n");
		free(result_tmp);
	}

	// for(int i = 0; i < amount; ++i){
	// 	if(result_tmp == NULL)
	// 		perror("Error Occure\n");
	// 	result_tmp = (int*)malloc(sizeof(int)*sizes[i]);
	// 	ASSERT_EQ(cudaMemcpy(result_tmp, result_arr[i], sizeof(int)*sizes[i], cudaMemcpyDeviceToHost), cudaSuccess) << "i = " << i<<std::endl;
	// 	qsort(values[i], sizes[i], sizeof(int), cmpint);
	// 	for(int j = 0; j < sizes[i]; ++j){
	// 		ASSERT_EQ(values[i][j], result_tmp[j]);
	// 	}
	// 	free(result_tmp);
	// }
}
