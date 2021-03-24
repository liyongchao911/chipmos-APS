#include <include/linked_list.h>
#include <stdio.h>
#undef DEBUG
#define show(list) \
	for(LinkedList *iter = list; iter; iter = iter->getNext()) { \
		printf("%.2f ", iter->getValue()); \
	} \
	printf("\n"); \

__device__ __host__ LinkedList::LinkedList(){
	this->next = this->prev = nullptr;
}

__device__ __host__ LinkedList * LinkedList::getNext(){
	return this->next;
}

__device__ __host__ LinkedList * LinkedList::getPrev(){
	return this->prev;
}

__device__ __host__ void LinkedList::setNext(LinkedList * next){
	this->next = next;
}

__device__ __host__ void LinkedList::setPrev(LinkedList * prev){
	this->prev = prev;
}

__device__ __host__ LinkedList * mergeLinkedList(LinkedList * l1, LinkedList * l2){
	if(!l2) return l1;
	if(!l1) return l2;
	
	LinkedList * result, *result_iter;
	
	// set the first element of result
	if(l1->getValue() < l2->getValue()) {
		result = l1;
		l1 = l1->next;	
	} else {
		result = l2;
		l2 = l2->next;
	}
	
	result_iter = result;
	
	// merge the linked list
	while(l1 && l2) {
		if(l1->getValue() < l2->getValue()){
			result_iter->next = l1; // connect to next element
			l1 = l1->next; // l1 move to next element
		} else{
			result_iter->next=  l2;
			l2 = l2->next; // l2 move to next element
		}
		result_iter = result_iter->next; // point to next element
	}
	
	// if l1 is not empty, connect to result
	if(l1) result_iter->next = l1;
	else if(l2) result_iter->next = l2;

	return result;

}


__device__ __host__ LinkedList * linkedListMergeSort(LinkedList * head){
	if(!head || !head->next) {
		return head;
	}else{
		LinkedList *fast = head->next;
		LinkedList *slow = head;
		
		// get the middle of linked list
		// divide the linked list
		while(fast && fast->next){
			slow = slow->next;
			fast = fast->next->next;
		}
		// now, get two lists.
		fast = slow->next;
		slow->next = NULL;
#ifdef DEBUG	
		printf("Head : ");
		show(head);
		printf("Fast : ");
		show(fast);
#endif
		LinkedList *lhs = linkedListMergeSort(head);
#ifdef DEBUG
		printf("lhs finish!\n");
#endif
		LinkedList *rhs = linkedListMergeSort(fast);
		
		// merge the linked list
		return mergeLinkedList(lhs, rhs);	
	}

}


// __global__ void vectorAddInt(int * a, int *b, int *c, unsigned int num_elements){
// 	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
// 	if(idx < num_elements){
// 		c[idx] = a[idx] + b[idx];
// 	}
// }

LinkedList::~LinkedList(){

}
