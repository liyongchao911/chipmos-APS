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

class LinkedList;

/** @class LinkedList
 *	@brief Class for linked list data structure.
 */

/**	@brief merge sort of LinkedList 
 * 	@return LinkedList* return the head of sorted list
 *
 * 	@param head head of a linked list
 */
__device__ __host__ LinkedList * linkedListMergeSort(LinkedList *head);

/**
 *
 */
__device__ __host__ LinkedList * mergeLinkedList(LinkedList * l1, LinkedList * l2);


class LinkedList{
friend class TestLinkedList;
	friend LinkedList * linkedListMergeSort(LinkedList*);
	friend LinkedList * mergeLinkedList(LinkedList*, LinkedList*);
private:
	LinkedList * next;
	LinkedList * prev;
public:
	/** @brief Construct LinkedList object 
	 */
	__device__ __host__ LinkedList();

	
	/** @brief Get the next LinkedList element
	 *
	 * 	@return LinkedList * or NULL
	 *	
	 *	@details
	 *	Get the next LinkedList element. If LinkedList element is not connected
	 *	to next element the function will return NULL.
	 */
	__device__ __host__ LinkedList * getNext();

	/**	@brief Get the previous LinkedList element
	 *	@return LinkedList * or NULL
	 *	@details
	 *	Get the previous LinkedList element. If LinkedList element is not connected
	 *	to previous element the function will return NULL.
	 */
	__device__ __host__ LinkedList * getPrev();

	/**	@brief Set the next element of the LinkedList
	 *	@return Void
	 *	@param next next element
	 *	@details
	 *	Connect to next LinkedList element.
	 */
	__device__ __host__ void setNext(LinkedList *next);

	/**	@brief Set the previous element of the LinkedList
	 * 	@return Void
	 * 	@param prev previous element
	 *
	 * 	@details
	 * 	Connect to previous LinkedList element
	 */
	__device__ __host__ void setPrev(LinkedList *);
	
	/**	@brief Get the value
	 * 	@return value the value of LinkedList
	 */
	virtual __device__ __host__ double getValue()=0;

	
	/**	@brief	Destruct the object
	 */
	virtual __device__ __host__ ~LinkedList();
};

// __global__ void vectorAddInt(int * a, int *b, int *c, unsigned int num_elements);

#endif
