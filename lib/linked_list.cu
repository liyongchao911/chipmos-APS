#include <include/linked_list.h>

__host__ LinkedList::LinkedList(){
	this->number = 0;
	this->next = this->prev = nullptr;
}

__host__ LinkedList::LinkedList(int number){
	this->number = number;
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

__device__ __host__ void LinkedList::setPrev(LinkedList * last){
	this->prev = last;
}

__device__ int LinkedList::getNumber(){
	return this->number;
}

__global__ void vectorAddInt(int * a, int *b, int *c, unsigned int num_elements){
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < num_elements){
		c[idx] = a[idx] + b[idx];
	}
}
