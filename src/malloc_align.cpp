#include "PCTracker.h"
#include "DFT.h"
#include "malloc_align.h"

void* MallocAlign( int nSize)
{
    unsigned char * buf;
    unsigned char * align_buf;
	int nAlign = 16;
	int tmp = nAlign - 1;
    buf = (unsigned char *) malloc( nSize + 15 + sizeof( void ** ) +
            sizeof( int ) );

//    buf = (unsigned char *) GlobalAlloc(GPTR , nSize + tmp + sizeof( void ** ) +
//              sizeof( int ) );
    align_buf = buf + tmp + sizeof( void ** ) + sizeof( int );
    align_buf -= (long) align_buf & tmp;
    *( (void **) ( align_buf - sizeof( void ** ) ) ) = buf;
    *( (int *) ( align_buf - sizeof( void ** ) - sizeof( int ) ) ) = nSize;
    return align_buf;
}

void FreeAlign(void *p)
{
//	if(p != NULL)
//		GlobalFree( (HGLOBAL)*( ( ( void **) p ) - 1 ) );
	if(p != NULL)
		free( *( ( ( void **) p ) - 1 ) );
}


