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
 * @brief LINKED_LIST_OPS will create default list_operations_t.
 * in which, the field setNext point to __listEleSetNext and the field setPrev point to __listEleSetPrev
 */
#define LINKED_LIST_OPS() list_operations_t{   \
	.setNext = __listEleSetNext,                        \
	.setPrev = __listEleSetPrev,                        \
}                                                       \



typedef struct list_ele_t list_ele_t;

typedef struct list_operations_t list_operations_t;

/**__listEleSetNext () - Default operation on struct list_ele_t.
 * The function is used to connect next list_ele_t. The two objects will be linked doubly.
 * @param _self : current list element
 * @param _next : next list element
 */
__device__ __host__ void __listEleSetNext(void *_self, list_ele_t *_next);

/**__listEleSetPrev () - Default operation on struct list_ele_t.
 * The function is used to connect previous list_ele_t. The two objects will be linked doubly.
 * @param _self : current list element
 * @param _prev : previous list element
 */
__device__ __host__ void __listEleSetPrev(void *_self, list_ele_t *_prev);


/**
 * initList () - Initialize the linked list element
 * In the function, the pointers of list element will be set NULL
 * @param _self : the element which is going to be initialized
 */
__device__ __host__ void initList(void *_self);


/**
 * linkedListMergeSort () - Sort the linked list by using merge sort algorithm
 * @param head: The head of linked list which is going to be sorted. The head is struct list_ele_t * type
 * @param ops: The basic operation on list element. ops should be struct list_ele_t * type.
 * @return head of list
 * @warning
 *  1. ops should not be NULL or the function will crash.
 *  2. The function will use getValue function pointer to evaluate each list_ele_t's value.
 *    Please make sure that getValue is not NULL and it point to the correct function before invoking this function.
 *  3. The function will use setNext function pointer to add new list node or concatenate two lists.
 *    Please make sure that setNext in @ops is point to correct function.
 *  */
__device__ __host__ list_ele_t * linkedListMergeSort(
        list_ele_t *head,
        list_operations_t *ops
                                                            );

/**
 * mergeLinkedList () - Merge two linked lists and return the result.
 * Merge two linked lists in increasing order.
 * @param l1: a list should be merged
 * @param l2: a list should be merged
 * @param ops: operation performed on list_ele_t object
 */
__device__ __host__ list_ele_t * mergeLinkedList(
        list_ele_t * l1,
        list_ele_t * l2,
        list_operations_t *ops
                                                        );

/**
 * newLinkedListElement() - create a list_ele_t object
 * The memory of object is allocated on heap. If fail on memory allocation the function return NULL or the function will will initialize the object and return.
 * @return NULL or list_ele_t object
 */
list_ele_t * newLinkedListElement();


/**
 * @struct list_ele_t
 * @brief A node of double-linked list
 *
 * Each node contains basically 4 pointers.
 * @b next point to next node of current node. @b prev point to previous node in linked list.
 * @b ptr_drived_object is a @b void* type pointer, which can be used to point to its parent object
 * The operations on list_ele_t object have a structure to maintain the function, the structure is LinkedListElementOpreation.
 * list_operations_t allows user to design their own operations on list_ele_t object.
 *
 * The list nodes are usually embedded in a container structure. @b ptr_derived_object is a @b void* type which can be used
 * to point to the address of container structure that help to manipulate structure.
 * @var next : pointer to next node in linked list
 * @var prev : pointer to previous node in linked list
 * @var ptr_derived_object : pointer to parent object
 */
struct list_ele_t{
    /// pointer to next node in linked list
	list_ele_t * next;
    
    /// pointer to previous node in linked list
	list_ele_t * prev;
    
    /// pointer to parent object
	void * ptr_derived_object;
    
    /// @brief function pointer to a function which is to evaluate the value of this node.
    /// The user must let the function pointer point to the correct function before invoking the function.
    double (*getValue)(void *self);
};


/**
 * @struct list_operations_t
 * @brief The structure to store all operations of struct list_ele_t. The user can define their own operations.
 *
 * The list_operations_t is used to link to correct functions on host or on device.
 * The address of device function and address of host function with the same name is different
 * but if we would like to make program flexble and maintainable
 * We need function pointer to point to which function we would like to perform on.
 * If we use use C++ like OOP,  it goes wrong because the traditional object orient programming
 * use VTable to make program flexble, but if heriented objects ard copied to device,
 * VTable is totally wrong because the pointed function is still on host instead of device.
 * That's why we need list_operations_t to store the correct function pointer.
 * The function pointer also gives user more flexibility to change the operation which will perform on list_ele_t object.
 *
 * @var init : pointer to a function to initialize the list_ele_t
 * @var rest : pointer to a function to reset the list_ele_t
 * @var setNext : pointer to function to set the next node of a list_ele_t
 * @var setPrev : pointer to function to set the previousnode of a list_ele_t
 */
struct list_operations_t{
    /// pointer to a function to initialize the list_ele_t
	void (*init)(void *self);
    
    /// pointer to a function to reset the list_ele_t
	void (*reset)(void *self);
    
    /// pointer to function to set the next node of a list_ele_t
	void (*setNext)(void *self, list_ele_t *next);
    
    /// pointer to function to set the previousnode of a list_ele_t
	void (*setPrev)(void *self, list_ele_t *prev);
};


#endif
