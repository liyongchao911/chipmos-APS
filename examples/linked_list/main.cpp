#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <include/linked_list.h>
#include <include/common.h>
#include "list_item.h"
#define amount 15

#define showList(iter, head)                                                 \
	iter = head;                                                             \
	while(iter) {printf("%.0f ", iter->getValue(iter)); iter = iter->next;}  \
	printf("\n");                                                            \

int main(int argc, const char *argv[]){
	
	double value[amount];
	for(int i = 0; i < amount; ++i){
		value[i] = rand() % 100;
	}

	LinkedListElementOperation ops = LINKED_LIST_OPS();
	
	LinkedListElement  *head = NULL;
	list_item_t *prev;
	for(int i = 0; i < amount; ++i){
		prev = new_list_item(value[i]);
		ops.setNext(&prev->ele, head);
		head = &prev->ele;
	}

	LinkedListElement *iter = head;
	printf("Unsorted : ");
	showList(iter, head);
	

	head = linkedListMergeSort(head, &ops);

	printf("Sorted : ");
	showList(iter, head);

	qsort(value, amount, sizeof(double), cmpdouble);
	iter = head;
	for(int i = 0; i < amount; ++i){
		assert(value[i] == iter->getValue(iter));
		iter = iter->next;
	}
}
