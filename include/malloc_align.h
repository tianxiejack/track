#ifndef _MALLOC_ALIGN_H
#define _MALLOC_ALIGN_H

#ifdef		__cplusplus
extern "C"{
#endif

void* MallocAlign( int nSize );

void FreeAlign(void *p);

#ifdef		__cplusplus
	}
#endif

#endif

