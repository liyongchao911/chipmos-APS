#include <include/linked_list.h>

__host__ LinkedList::LinkedList(){
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

	if(l1->getValue() < l2->getValue()) {
		l1->setNext(mergeLinkedList(l1->getNext(), l2));
		return l1;
	} else {
		l2->setNext(mergeLinkedList(l1, l2->getNext()));
		return l2;
	}
}


__device__ __host__ LinkedList * linkedListMergeSort(LinkedList * head){
	if(!head && !head->getNext()) return head;

	LinkedList *fast = head->getNext();
	LinkedList *slow = head;
	while(fast && fast->getNext()){
		slow = slow->getNext();
		fast = fast->getNext()->getNext();
	}
	fast = slow->getNext();
	slow->setNext(NULL);

	LinkedList *lhs = linkedListMergeSort(head);
	LinkedList *rhs = linkedListMergeSort(slow);

	return mergeLinkedList(lhs, rhs);	

}


__global__ void vectorAddInt(int * a, int *b, int *c, unsigned int num_elements){
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < num_elements){
		c[idx] = a[idx] + b[idx];
	}
}

LinkedList::~LinkedList(){

}
