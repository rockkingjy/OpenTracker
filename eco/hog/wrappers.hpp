/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.00
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#ifndef _WRAPPERS_HPP_
#define _WRAPPERS_HPP_

//#include <stdio.h>
//#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)

//#include <stddef.h>
#include <stdlib.h>
//#include <assert.h>

// wrapper functions if compiling from C/C++
inline void wrError(const char *errormsg) { throw errormsg; }
inline void *wrCalloc(size_t num, size_t size) { return calloc(num, size); }
inline void *wrMalloc(size_t size) { return malloc(size); }
inline void wrFree(void *ptr) { free(ptr); }


// platform independent aligned memory allocation (see also alFree)
// __m128 should be 128/8=16byte aligned
inline void *alMalloc(size_t size, int alignment)
{
  const size_t pSize = sizeof(void *), a = alignment - 1;
  void *raw = wrMalloc(size + a + pSize);
  // get the aligned address, allignment should be 2^N.
  void *aligned = (void *)(((size_t)raw + pSize + a) & ~a); 
  *(void **)((size_t)aligned - pSize) = raw; // save address of raw in -1
  //debug("malloc: %lu, aligned: %lu, psize: %lu", raw, aligned, pSize);
  return aligned;
}

// platform independent alignned memory de-allocation (see also alMalloc)
inline void alFree(void *aligned)
{
  // raw: the address of (void *) pointer.
  // aligned: the address of a (void *) pointer now point to (char *)
  // - sizeof(void *): minus the address by sizeof(void *)
  // (void **): the address of a pointer point to a (void *) pointer
  // *: the pointer point to a (void *) pointer = the address of (void *)pointer
  void *raw = *(void **)((char *)aligned - sizeof(void *));
  //debug("aligned:%lu, raw:%lu, psize:%lu", aligned, raw, sizeof(void *));
  wrFree(raw);
}

/*
//https://zhoujianshi.github.io/articles/2017/任意字节对齐的动态内存分配函数/index.html

#define PTR_ADDR(p) ((unsigned long)p)

inline void* alMalloc(uint32_t size,uint8_t alignment)
{
    assert(alignment>0);
    uint8_t* bytes=(uint8_t*)malloc(size+alignment);
    uint8_t offset=alignment-PTR_ADDR(bytes)%alignment;
    bytes+=offset;
    bytes[-1]=offset;
    return bytes;
}

inline void alFree(void* ptr)
{
    assert(ptr!=0);
    uint8_t* bytes=(uint8_t*)ptr;
    uint8_t offset=bytes[-1];
    bytes-=offset;
    free(bytes);
}
*/
#endif
