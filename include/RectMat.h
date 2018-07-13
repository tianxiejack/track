#ifndef		_RECT_MAT_H_
#define		_RECT_MAT_H_

#include "DFT.h"

typedef struct _rect_i_{
	int x, y;
	int width, height;
}Recti;

typedef struct _rect_f_{
	float x, y;
	float width, height;
}Rectf;

typedef struct _size_t_{
	int  width, height;
}Sizei;

void ResizeMat(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst);

void ResizeMat_interp(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst);

void SubWindowMat(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst, Recti window, Sizei tmpl_sz);

void GradXY(IMG_MAT_UCHAR src, IMG_MAT_FLOAT *pdst);

void CutWindowMat(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst, Recti window);

void _IMG_sobel(  IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst) ;


#endif
