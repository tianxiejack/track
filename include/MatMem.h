#ifndef		_MAT_MEM_HEAD
#define		_MAT_MEM_HEAD

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <math.h>

#include "PCTracker.h"
#include "DFTTrack.h"
#include "DFT.h"
#include "malloc_align.h"

IMG_MAT *matAlloc(int dtype, int width, int height, int channels);

void matFree(IMG_MAT *mat);

IMG_MAT* matCopy(IMG_MAT *dst, IMG_MAT *src);

IMG_MAT* matMul(IMG_MAT *dst, IMG_MAT *src);

IMG_MAT *transpose(IMG_MAT *mat);

#endif
