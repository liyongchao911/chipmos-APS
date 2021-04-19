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

__device__ __host__ void __listEleSetNext(void *_self, list_ele_t *_next){
	list_ele_t * self = (list_ele_t *)_self;
	self->next = _next;
	_next->prev = self;
}


__device__ __host__ void __listEleSetPrev(void *_self, list_ele_t *_prev){
	list_ele_t *self = (list_ele_t*)_self;
	self->prev = _prev;
	_prev->next = self;
}

__device__ __host__ void initList(void *_self){
	list_ele_t *self	= (list_ele_t *)_self;
	self->next = self->prev = NULL;
	self->getValue = NULL;
}

list_ele_t * newLinkedListElement(){
	list_ele_t * ele = (list_ele_t*)malloc(sizeof(list_ele_t));
	if(!ele)
		return ele;
	ele->ptr_derived_object = NULL;
    ele->next = ele->prev = NULL;
	return ele;
}

__device__ __host__ list_ele_t * mergeLinkedList(list_ele_t * l1, list_ele_t * l2, list_operations_t *ops){
	if(!l2) return l1;
	if(!l1) return l2;

	list_ele_t * result, *result_iter;
	
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


__device__ __host__  list_ele_t * linkedListMergeSort(list_ele_t * head, list_operations_t *ops){
	if(!head || !head->next) {
		return head;
	}else{
		
		list_ele_t *fast = (list_ele_t*)head->next;
		list_ele_t *slow = head;
		
		// get the middle of linked list
		// divide the linked list
		while(fast && fast->next){
			slow = (list_ele_t*)slow->next;
			fast = (list_ele_t*)((list_ele_t*)fast->next)->next;
		}
		// now, get two lists.
		fast = (list_ele_t*)slow->next;
		fast->prev = NULL;
		slow->next = NULL;
#ifdef DEBUG	
		printf("Head : ");
		show(head);
		printf("Fast : ");
		show(fast);
#endif
		list_ele_t *lhs = linkedListMergeSort(head, ops);
#ifdef DEBUG
		printf("lhs finish!\n");
#endif
		list_ele_t *rhs = linkedListMergeSort(fast, ops);
		
		// merge the linked list
#ifdef DEBUG
		list_ele_t *result = mergeLinkedList(lhs, rhs, ops);
		printf("Merge Result : ");
		show(result);
		return result;
#endif

		return mergeLinkedList(lhs, rhs, ops);
	}

}
