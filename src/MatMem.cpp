#include "MatMem.h"

#ifndef min
#  define min(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef max
#  define max(a,b)  ((a) < (b) ? (b) : (a))
#endif

/*********/

IMG_MAT *matAlloc(int dtype, int width, int height, int channels)
{
	IMG_MAT *mat = NULL;
	UTILS_assert(width>0 && height>0 && channels>0);
	UTILS_assert(dtype == 0 || dtype == 1);

	mat = (IMG_MAT *)malloc(sizeof(IMG_MAT));
	UTILS_assert(mat != NULL);
	memset(mat, 0, sizeof(IMG_MAT));

	if(dtype == MAT_u8)
	{
		mat->size = width*height*channels;
		mat->data_u8 = (unsigned char*)MallocAlign(mat->size);
		UTILS_assert(mat->data_u8 != NULL);
	}
	else
	{
		mat->size = width*height*channels*sizeof(float);
		mat->data = (float*)MallocAlign(mat->size);
		UTILS_assert(mat->data != NULL);
	}
	mat->width = width;
	mat->height = height;
	mat->channels = channels;
	mat->step[0] = width*channels;
	mat->dtype = dtype;

	memset(mat->data_u8, 0, mat->size);

	return (mat);
}

void matFree(IMG_MAT *mat)
{
	if(mat == NULL)
		return;

	if(mat->data_u8 != NULL)
		FreeAlign(mat->data_u8);

	free(mat);
}

IMG_MAT* matCopy(IMG_MAT *dst, IMG_MAT *src)
{
	int i;
	UTILS_assert(dst != NULL && src != NULL);
	if(dst != src){
		if(src->data_u8 != NULL){
			if(dst->data_u8 == NULL || dst->size<src->size)
			{
				if(dst->data_u8 != NULL)
					FreeAlign(dst->data_u8);
				dst->size = src->size;
				dst->data_u8 = (unsigned char*)MallocAlign(dst->size);
				UTILS_assert(dst->data_u8 != NULL);
			}
			memcpy(dst->data_u8, src->data_u8, src->size);
		}
		dst->width = src->width;
		dst->height = src->height;
		dst->channels = src->channels;
		for(i=0; i<MAX_DIM; i++)
			dst->step[i] = src->step[i];
		dst->dtype = src->dtype;
	}
	return dst;
}

IMG_MAT* matMul(IMG_MAT *dst, IMG_MAT *src)
{
	int i;
	float *pdst, *psrc;
	UTILS_assert(dst != NULL);
	UTILS_assert(src != NULL);
	UTILS_assert(dst->width == src->width && dst->height == src->height);
	UTILS_assert(dst->channels == src->channels);
	UTILS_assert(dst->channels == 1);
	UTILS_assert(dst->dtype == MAT_float);

	psrc = src->data;
	pdst = dst->data;
	#pragma UNROLL(4)
	for(i=0; i<dst->height*dst->width; i++){
		pdst[i] = pdst[i] * psrc[i];
	}

	return dst;
}

__INLINE__ void transpose_(
	const unsigned char* src,
	int sstep,
	unsigned char* dst,
	int dstep,
	int swidth,
	int sheight )
{
	int i=0, j, m = swidth, n = sheight;
#define T float
#if 1
	for(; i <= m - 4; i += 4 )
	{
		T* d0 = (T*)(dst + dstep*i);
		T* d1 = (T*)(dst + dstep*(i+1));
		T* d2 = (T*)(dst + dstep*(i+2));
		T* d3 = (T*)(dst + dstep*(i+3));

		for( j = 0; j <= n - 4; j += 4 )
		{
			const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
			const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
			const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
			const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

			d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
			d1[j] = s0[1]; d1[j+1] = s1[1]; d1[j+2] = s2[1]; d1[j+3] = s3[1];
			d2[j] = s0[2]; d2[j+1] = s1[2]; d2[j+2] = s2[2]; d2[j+3] = s3[2];
			d3[j] = s0[3]; d3[j+1] = s1[3]; d3[j+2] = s2[3]; d3[j+3] = s3[3];
		}

		for( ; j < n; j++ )
		{
			const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
			d0[j] = s0[0]; d1[j] = s0[1]; d2[j] = s0[2]; d3[j] = s0[3];
		}
	}
#endif
	for( ; i < m; i++ )
	{
		T* d0 = (T*)(dst + dstep*i);
		j = 0;
#if 1
		for(; j <= n - 4; j += 4 )
		{
			const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
			const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
			const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
			const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

			d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
		}
#endif
		for( ; j < n; j++ )
		{
			const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
			d0[j] = s0[0];
		}
	}
#undef T
}

IMG_MAT *transpose(IMG_MAT *mat)
{
	unsigned char *data = (unsigned char *)MallocAlign(mat->size);
	int width = mat->width;
	int height = mat->height;

	UTILS_assert(data != NULL);

	transpose_(mat->data_u8,
		mat->width*mat->channels*sizeof(float),
		data,
		mat->height*mat->channels*sizeof(float),
		mat->width, mat->height);

	FreeAlign(mat->data);
	mat->data_u8 = data;
	mat->width = height;
	mat->height = width;
	mat->step[0] = mat->width*mat->channels;

	return (mat);
}
