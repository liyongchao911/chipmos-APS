#include "include/linked_list.h"
#include <tests/include/test_linked_list.h>

__device__ __host__ double linkedListItemGetValue(void *_self){
	list_ele_t * ele = (list_ele_t*)_self;
	if(ele){
		list_item_t *item = (list_item_t *)ele->ptr_derived_object;
		return item->value;
	}
	return 0;
}

list_item_t * new_list_item(double val){
	list_item_t *item = (list_item_t *)malloc(sizeof(list_item_t));
	if(!item)
		return NULL;
	// item->ele = newLinkedListElement();
	// item->ele.setNext = 
	initList(&(item->ele));
	// item->ele.setNext = __listEleSetNext;
	// item->ele.setPrev = __listEleSetPrev;
	item->ele.ptr_derived_object = item;
	item->ele.getValue = linkedListItemGetValue;
	item->value = val;
	return item;
}

void list_item_add(list_item_t ** list, list_item_t *item, list_operations_t ops){
	if(!item){
		perror("ERROR : item is NULL\n");
		return;	
	}
	if(*list){
		ops.setNext(&(item->ele), &(*list)->ele);
		// item->ele.setNext(&(item->ele), &(*list)->ele);
		*list = item;	
	}else{
		*list = item;
	}
}

