//
// Created by eugene on 2021/4/30.
//

#ifndef __DEF_H__
#define __DEF_H__

#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>

#define __qualifier__ __host__ __device__
#else
#define __qualifier__
#endif

#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

#if !defined(__GNUC__)
#define __builtin_clz __lzcnt
#endif

#endif
