#ifndef __TEST_LINKED_LIST_H__
#define __TEST_LINKED_LIST_H__

#include <include/linked_list.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct LinkedListItem LinkedListItem;
struct LinkedListItem {
	LinkedListElement ele;
	double value;
};

__device__ __host__ double linkedListItemGetValue(void *_self);

LinkedListItem * newLinkedListItem(double val);


void LinkedListItemAdd( LinkedListItem ** list, LinkedListItem *item);

int cmpint(const void *a, const void *b);

#endif
