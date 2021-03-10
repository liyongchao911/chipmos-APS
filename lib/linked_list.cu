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
	
	LinkedList * result, *result_iter;
	
	// set the first element of result
	if(l1->getValue() < l2->getValue()) {
		result = l1;
		l1 = l1->getNext();	
	} else {
		result = l2;
		l2 = l2->getNext();
	}
	
	result_iter = result;
	
	// merge the linked list
	while(l1 && l2) {
		if(l1->getValue() < l2->getValue()){
			result_iter->setNext(l1); // connect to next element
			l1 = l1->getNext(); // l1 move to next element
		} else{
			result_iter->setNext(l2);
			l2 = l2->getNext(); // l2 move to next element
		}
		result_iter = result_iter->getNext(); // point to next element
	}
	
	// if l1 is not empty, connect to result
	if(l1) result_iter->setNext(l1);
	else if(l2) result_iter->setNext(l2);

	return result;

}


__device__ __host__ LinkedList * linkedListMergeSort(LinkedList * head){
	if(!head && !head->getNext()) return head;

	LinkedList *fast = head->getNext();
	LinkedList *slow = head;
	
	// get the middle of linked list
	// divide the linked list
	while(fast && fast->getNext()){
		slow = slow->getNext();
		fast = fast->getNext()->getNext();
	}
	// now, get two lists.
	fast = slow->getNext();
	slow->setNext(NULL);
	

	LinkedList *lhs = linkedListMergeSort(head);
	LinkedList *rhs = linkedListMergeSort(slow);
	
	// merge the linked list
	return mergeLinkedList(lhs, rhs);	

}


// __global__ void vectorAddInt(int * a, int *b, int *c, unsigned int num_elements){
// 	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
// 	if(idx < num_elements){
// 		c[idx] = a[idx] + b[idx];
// 	}
// }

LinkedList::~LinkedList(){

}
