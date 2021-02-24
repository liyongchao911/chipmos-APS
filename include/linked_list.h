#ifndef __LINKED_LIST_H__
#define __LINKED_LIST_H__

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

class LinkedList{
friend class TestLinkedList;
private:
	LinkedList * next;
	LinkedList * prev;
	int number;
public:
	__host__ LinkedList();
	__host__ LinkedList(int number);
	__device__ __host__ LinkedList * getNext();
	__device__ __host__ LinkedList * getPrev();
	__device__ __host__ void setNext(LinkedList *);
	__device__ __host__ void setPrev(LinkedList *);
	virtual __device__ __host__ int getNumber();
	virtual ~LinkedList();
};
__global__ void vectorAddInt(int * a, int *b, int *c, unsigned int num_elements);

#endif
