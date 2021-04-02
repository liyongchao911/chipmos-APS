#include "test_linked_list.h"

__device__ __host__ double linkedListItemGetValue(void *_self){
	LinkedListElement * ele = (LinkedListElement*)_self;
	if(ele){
		LinkedListItem *item = (LinkedListItem *)ele->pDerivedObject;
		return item->value;
	}
	return 0;
}

LinkedListItem * newLinkedListItem(double val){
	LinkedListItem *item = (LinkedListItem *)malloc(sizeof(LinkedListItem));
	// item->ele = newLinkedListElement();
	// item->ele.setNext = 
	initList(&(item->ele));
	item->ele.setNext = __listEleSetNext;
	item->ele.setPrev = __listEleSetPrev;
	item->ele.pDerivedObject = item;
	item->ele.getValue = linkedListItemGetValue;
	item->value = val;
	return item;
}

void LinkedListItemAdd( LinkedListItem ** list, LinkedListItem *item){
	if(!item){
		perror("ERROR : item is NULL\n");
		return;	
	}
	if(*list){
		item->ele.setNext(&(item->ele), &(*list)->ele);
		*list = item;	
	}else{
		*list = item;
	}
}

int cmpint(const void *a, const void *b){
	return *(int*)a > *(int*)b;
}
