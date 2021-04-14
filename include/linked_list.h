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

/**
 * @def LINKED_LIST_OPS
 * @brief LINKED_LIST_OPS will create default LinkedListElementOperation.
 * in which, the field setNext point to __listEleSetNext and the field setPrev point to __listEleSetPrev
 */
#define LINKED_LIST_OPS() LinkedListElementOperation{   \
	.setNext = __listEleSetNext,                        \
	.setPrev = __listEleSetPrev,                        \
}                                                       \



typedef struct LinkedListElement LinkedListElement;

typedef struct LinkedListElementOperation LinkedListElementOperation;

/**__listEleSetNext () - Default operation on struct LinkedListElement.
 * The function is used to connect next LinkedListElement. The two objects will be linked doubly.
 * @param _self : current list element
 * @param _next : next list element
 */
__device__ __host__ void __listEleSetNext(void *_self, LinkedListElement *_next);

/**__listEleSetPrev () - Default operation on struct LinkedListElement.
 * The function is used to connect previous LinkedListElement. The two objects will be linked doubly.
 * @param _self : current list element
 * @param _prev : previous list element
 */
__device__ __host__ void __listEleSetPrev(void *_self, LinkedListElement *_prev);


/**
 * initList () - Initialize the linked list element
 * In the function, the pointers of list element will be set NULL
 * @param _self : the element which is going to be initialized
 */
__device__ __host__ void initList(void *_self);


/**
 * linkedListMergeSort () - Sort the linked list by using merge sort algorithm
 * @param head: The head of linked list which is going to be sorted. The head is struct LinkedListElement * type
 * @param ops: The basic operation on list element. ops should be struct LinkedListElement * type.
 * @return head of list
 * @warning
 *  1. ops should not be NULL or the function will crash.
 *  2. The function will use getValue function pointer to evaluate each LinkedListElement's value.
 *    Please make sure that getValue is not NULL and it point to the correct function before invoking this function.
 *  3. The function will use setNext function pointer to add new list node or concatenate two lists.
 *    Please make sure that setNext in @ops is point to correct function.
 *  */
__device__ __host__ LinkedListElement * linkedListMergeSort(
                                                            LinkedListElement *head,
                                                            LinkedListElementOperation *ops
                                                            );

/**
 * mergeLinkedList () - Merge two linked lists and return the result.
 * Merge two linked lists in increasing order.
 * @param l1: a list should be merged
 * @param l2: a list should be merged
 * @param ops: operation performed on LinkedListElement object
 */
__device__ __host__ LinkedListElement * mergeLinkedList(
                                                        LinkedListElement * l1,
                                                        LinkedListElement * l2,
                                                        LinkedListElementOperation *ops
                                                        );

/**
 * newLinkedListElement() - create a LinkedListElement object
 * The memory of object is allocated on heap. If fail on memory allocation the function return NULL or the function will will initialize the object and return.
 * @return NULL or LinkedListElement object
 */
LinkedListElement * newLinkedListElement();


/**
 * @struct LinkedListElement
 * @brief A node of double-linked list
 *
 * Each node contains basically 4 pointers.
 * @b next point to next node of current node. @b prev point to previous node in linked list.
 * @b ptr_drived_object is a @b void* type pointer, which can be used to point to its parent object
 * The operations on LinkedListElement object have a structure to maintain the function, the structure is LinkedListElementOpreation.
 * LinkedListElementOperation allows user to design their own operations on LinkedListElement object.
 *
 * The list nodes are usually embedded in a container structure. @b ptr_derived_object is a @b void* type which can be used
 * to point to the address of container structure that help to manipulate structure.
 * @var next : pointer to next node in linked list
 * @var prev : pointer to previous node in linked list
 * @var ptr_derived_object : pointer to parent object
 */
struct LinkedListElement{
    /// pointer to next node in linked list
	LinkedListElement * next;
    
    /// pointer to previous node in linked list
	LinkedListElement * prev;
    
    /// pointer to parent object
	void * ptr_derived_object;
    
    /// @brief function pointer to a function which is to evaluate the value of this node.
    /// The user must let the function pointer point to the correct function before invoking the function.
    double (*getValue)(void *self);
};


/**
 * @struct LinkedListElementOperation
 * @brief The structure to store all operations of struct LinkedListElement. The user can define their own operations.
 *
 * The LinkedListElementOperation is used to link to correct functions on host or on device.
 * The address of device function and address of host function with the same name is different
 * but if we would like to make program flexble and maintainable
 * We need function pointer to point to which function we would like to perform on.
 * If we use use C++ like OOP,  it goes wrong because the traditional object orient programming
 * use VTable to make program flexble, but if heriented objects ard copied to device,
 * VTable is totally wrong because the pointed function is still on host instead of device.
 * That's why we need LinkedListElementOperation to store the correct function pointer.
 * The function pointer also gives user more flexibility to change the operation which will perform on LinkedListElement object.
 *
 * @var init : pointer to a function to initialize the LinkedListElement
 * @var rest : pointer to a function to reset the LinkedListElement
 * @var setNext : pointer to function to set the next node of a LinkedListElement
 * @var setPrev : pointer to function to set the previousnode of a LinkedListElement
 */
struct LinkedListElementOperation{
    /// pointer to a function to initialize the LinkedListElement
	void (*init)(void *self);
    
    /// pointer to a function to reset the LinkedListElement
	void (*reset)(void *self);
    
    /// pointer to function to set the next node of a LinkedListElement
	void (*setNext)(void *self, LinkedListElement *next);
    
    /// pointer to function to set the previousnode of a LinkedListElement
	void (*setPrev)(void *self, LinkedListElement *prev);
};


#endif
