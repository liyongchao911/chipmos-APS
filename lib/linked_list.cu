#include <features.h>
#include <include/linked_list.h>
#include <stdlib.h>
#include <stdio.h>
// #define DEBUG
#define show(list) 													\
   for(LinkedListElement *iter = list; iter; iter = iter->next) {    \
		printf("%.2f ", iter->getValue(iter));  						\
	}  																\
	printf("\n"); 													\

__device__ __host__ void __listEleSetNext(void *_self, LinkedListElement *_next){
	LinkedListElement * self = (LinkedListElement *)_self;	
	self->next = _next;
	_next->prev = self;
}


__device__ __host__ void __listEleSetPrev(void *_self, LinkedListElement *_prev){
	LinkedListElement *self = (LinkedListElement*)_self;
	self->prev = _prev;
	_prev->next = self;
}

__device__ __host__ void initList(void *_self){
	LinkedListElement *self	= (LinkedListElement *)_self;
	self->next = self->prev = NULL;
	// self->reset = NULL;
	self->getValue = NULL;
	// self->setNext = __listEleSetNext;
	// self->setPrev = __listEleSetPrev;

}

LinkedListElement * newLinkedListElement(){
	LinkedListElement * ele = (LinkedListElement*)malloc(sizeof(LinkedListElement));
	if(!ele)
		return ele;
	ele->ptr_derived_object = NULL;
	// ele->init = initList;
	// ele->init(ele);
	return ele;
}

__device__ __host__ LinkedListElement * mergeLinkedList(LinkedListElement * l1, LinkedListElement * l2, LinkedListElementOperation *ops){
	if(!l2) return l1;
	if(!l1) return l2;

	LinkedListElement * result, *result_iter;
	
	// set the first element of result
	if(l1->getValue(l1) < l2->getValue(l2)) {
		result = l1;
		l1 = l1->next;
	} else {
		result = l2;
		l2 = l2->next;
	}
	
	result_iter = result;
	
	// merge the linked list
	while(l1 && l2) {
		if(l1->getValue(l1) < l2->getValue(l2)){
			// result_iter->next = l1; // connect to next element
			__listEleSetNext(result_iter, l1);
			// ops.setNext(result_iter, l1);
			// result_iter->setNext(result_iter, l1);
			l1 = l1->next; // l1 move to next element
		} else{
			// result_iter->next = l2;
			// ops.setNext(result_iter, l2);
			// result_iter->setNext(result_iter, l2);
			__listEleSetNext(result_iter, l2);
			l2 = l2->next; // l2 move to next element
		}
		result_iter = result_iter->next; // point to next element
	}
	
	// if l1 is not empty, connect to result
	// if(l1) result_iter->setNext(result_iter, l1);
	// else if(l2) result_iter->setNext(result_iter, l2);
	// if(l1) ops.setNext(result_iter, l1);
	// else if(l2) ops.setNext(result_iter, l2);
	if(l1) __listEleSetNext(result_iter, l1);
	else if(l2) __listEleSetNext(result_iter, l2);
	return result;

}


__device__ __host__  LinkedListElement * linkedListMergeSort(LinkedListElement * head, LinkedListElementOperation *ops){
	if(!head || !head->next) {
		return head;
	}else{
		
		LinkedListElement *fast = (LinkedListElement*)head->next;
		LinkedListElement *slow = head;
		
		// get the middle of linked list
		// divide the linked list
		while(fast && fast->next){
			slow = (LinkedListElement*)slow->next;
			fast = (LinkedListElement*)((LinkedListElement*)fast->next)->next;
		}
		// now, get two lists.
		fast = (LinkedListElement*)slow->next;
		fast->prev = NULL;
		slow->next = NULL;
#ifdef DEBUG	
		printf("Head : ");
		show(head);
		printf("Fast : ");
		show(fast);
#endif
		LinkedListElement *lhs = linkedListMergeSort(head, ops);
#ifdef DEBUG
		printf("lhs finish!\n");
#endif
		LinkedListElement *rhs = linkedListMergeSort(fast, ops);
		
		// merge the linked list
#ifdef DEBUG
		LinkedListElement *result = mergeLinkedList(lhs, rhs, ops);
		printf("Merge Result : ");
		show(result);
		return result;
#endif

		return mergeLinkedList(lhs, rhs, ops);
	}

}
