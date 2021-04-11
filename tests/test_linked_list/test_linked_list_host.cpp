#include <include/linked_list.h>
#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <texture_types.h>
#include <iostream>
#include "test_linked_list.h"
#include <tests/def.h>

#undef amount
#define amount 5000

using namespace std;



class TestLinkedListHost : public testing::Test{
public:
	LinkedListElement ** eles;
	LinkedListItem ** eles_arr;
	int values[amount][amount];
	int sizes[amount];
	
	void SetUp() override;
	void TearDown() override;
};

void TestLinkedListHost::SetUp(){
	eles = (LinkedListElement**)malloc(sizeof(LinkedListElement *)*amount);
	for(int i = 0; i < amount; ++i){
		eles[i] = newLinkedListElement();
	}

	eles_arr = (LinkedListItem **)calloc(amount, sizeof(LinkedListItem *));
	int nums, val;
	int count = 0;	
	for(int i = 0; i < amount; ++i){
		nums = rand() % 10;
		sizes[i] = nums;
		for(int j = 0; j < nums; ++j){
			val = rand() % 1024;
			LinkedListItemAdd(&eles_arr[i], newLinkedListItem(val), LINKED_LIST_OPS());
			values[i][j] = val;
			count ++;
		}
	}
	PRINTF("Amount of Linked list is %d\n", count);
}

void TestLinkedListHost::TearDown(){
	if(eles){
		for(int i = 0; i < amount; ++i)
			free(eles[i]);
		// free(eles);
		eles = NULL;
	}
		
	// LinkedListItem *p;
	// LinkedListElement *n;
	// if(eles_arr){
	// 	for(int i = 0; i < amount; ++i){
	// 		for(n = &eles_arr[i]->ele; n;){
	// 			p = (LinkedListItem *)n->pDerivedObject;
	// 			n = n->next;
	// 			free(p);
	// 		}
	// 	}
	// 	// free(eles_arr);
	// 	eles_arr = NULL;

	// }
}


// Test 
TEST_F(TestLinkedListHost, test_set_next_on_host){
	LinkedListElementOperation ops = LINKED_LIST_OPS();
	for(int i = 0, range = amount - 1; i < range; ++i){
		ops.setNext(eles[i], eles[i+1]);
		// eles[i]->setNext(eles[i], eles[i + 1]);
	}

	for(int i = 0, range = amount - 1; i < range; ++i){
		ASSERT_EQ(eles[i]->next, eles[i + 1]);
	}

	for(int i = 1; i < amount; ++i){
		ASSERT_EQ(eles[i]->prev, eles[i - 1]);
	}
}

TEST_F(TestLinkedListHost, test_set_prev_on_host){
	LinkedListElementOperation ops = LINKED_LIST_OPS();
	for(int i = 1; i < amount; ++i){
		ops.setPrev(eles[i], eles[i-1]);
		// eles[i]->setPrev(eles[i], eles[i - 1]);
	}

	for(int i = 1; i < amount; ++i){
		ASSERT_EQ(eles[i]->prev, eles[i - 1]);
	}

	for(int i = 0, range = amount - 1; i < range; ++i){
		ASSERT_EQ(eles[i]->next, eles[i + 1]);
	}
}

TEST_F(TestLinkedListHost, test_sort_linked_list_on_host){
	LinkedListElement * iter;
	LinkedListElementOperation ops = LINKED_LIST_OPS();
	// LinkedListElement test;
	for(int i = 0; i < amount; ++i){
		qsort(values[i], sizes[i], sizeof(int), cmpint);
		if(sizes[i] != 0){
			iter = linkedListMergeSort(&(eles_arr[i]->ele), &ops);
			eles_arr[i] = (LinkedListItem *)iter->ptr_derived_object;
			iter = &eles_arr[i]->ele;
			// printf("Value : ");
			for(int j = 0; j < sizes[i]; ++j){
				// printf("%.2f ", iter->getValue(iter));
				ASSERT_EQ(iter->getValue(iter), values[i][j]);
				iter = iter->next;
			}
		}
	}
}
