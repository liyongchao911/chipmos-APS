/** 
 * @file linked_list.h
 * @brief linked list definition
 * @author Eugene Lin <lin.eugene.l.e@gmail.com>
 * @date 2021.2.23
 */
#ifndef __LINKED_LIST_H__
#define __LINKED_LIST_H__

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define LINKED_LIST_OPS() LinkedListElementOperation{ 				    \
	.setNext = __listEleSetNext, 										\
	.setPrev = __listEleSetPrev, 										\
} 																		\



typedef struct LinkedListElement LinkedListElement;

typedef struct LinkedListElementOperation LinkedListElementOperation;


__device__ __host__ void __listEleSetNext(void *_self, LinkedListElement *_next);

__device__ __host__ void __listEleSetPrev(void *_self, LinkedListElement *_next);

__device__ __host__ void initList(void *_self);

/** @class LinkedList
 *	@brief Class for linked list data structure.
 */

/**	@brief merge sort of LinkedList 
 * 	@return LinkedList* return the head of sorted list
 *
 * 	@param head head of a linked list
 */
__device__ __host__ LinkedListElement * linkedListMergeSort(LinkedListElement *head, LinkedListElementOperation *ops);

/**
 *
 */
__device__ __host__ LinkedListElement * mergeLinkedList(LinkedListElement * l1, LinkedListElement * l2, LinkedListElementOperation *ops);


LinkedListElement * newLinkedListElement();

struct LinkedListElement{
	
	LinkedListElement * next;
	LinkedListElement * prev;
	void * ptr_derived_object;
	
	/**	@brief Get the value
	 * 	@detail getValue is an
	 * 	@return value the value of LinkedList
	 */
	double (*getValue)(void *self);
};

struct LinkedListElementOperation{
	void (*init)(void *self);
	void (*reset)(void *self);
	void (*setNext)(void *self, LinkedListElement *next);
	void (*setPrev)(void *self, LinkedListElement *prev);
};


#endif
