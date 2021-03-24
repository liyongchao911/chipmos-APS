#ifndef __TEST_LINKED_LIST_H__
#define __TEST_LINKED_LIST_H__

#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <include/linked_list.h>
#include <vector>
#include <gtest/gtest.h>

class LinkedListChild : public LinkedList{
private:
	double value;
public:
	__device__ __host__ LinkedListChild() : LinkedList(){
		value = 0;
	}
	__device__ __host__ LinkedListChild(double val):LinkedList(){
		value = val;	
	}

	__device__ __host__ void reset(){
		value = 0;
	}

	__device__ __host__ void setValue(double val){
		value = val;
	}

	virtual __device__ __host__ double getValue(){
		return value;
	}
};

#endif
