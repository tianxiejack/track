#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "PCTracker.h"
#include "RectMat.h"
#include "neon_std.h"

static void Resize_cubic(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst);
static void Resize_cubic2(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst);

__INLINE__ int x2(const Recti rect)
{
	return rect.x + rect.width;
}

__INLINE__ int y2(const Recti rect)
{
	return rect.y + rect.height;
}

__INLINE__ Recti resize(Recti rect, float scalex, float scaley)
{
	Recti	rslt;
	if (!scaley)	
		scaley = scalex;
	rect.x -= rect.width * (scalex - 1.f) / 2.f;
	rect.width *= scalex;

	rect.y -= rect.height * (scaley - 1.f) / 2.f;
	rect.height *= scaley;
	rslt = rect;
	return rslt;
}

__INLINE__ Recti limit_(Recti rect, Recti limit)
{
	Recti	limRec;
	if (rect.x + rect.width > limit.x + limit.width)		rect.width = (limit.x + limit.width - rect.x);
	if (rect.y + rect.height > limit.y + limit.height)	rect.height = (limit.y + limit.height - rect.y);
	if (rect.x < limit.x)
	{
		rect.width -= (limit.x - rect.x);
		rect.x = limit.x;
	}
	if (rect.y < limit.y)
	{
		rect.height -= (limit.y - rect.y);
		rect.y = limit.y;
	}
	if(rect.width<0)		rect.width=0;
	if(rect.height<0)	rect.height=0;

	limRec = rect;
	return limRec;
}

__INLINE__ Recti getBorder(const Recti original, Recti limited)
{
	Recti res;
	res.x = limited.x - original.x;
	res.y = limited.y - original.y;
	res.width = x2(original) - x2(limited);
	res.height = y2(original) - y2(limited);
	UTILS_assert(res.x >= 0 && res.y >= 0 && res.width >= 0 && res.height >= 0);
	return res;
}

void SubWindowMat(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst, Recti window, Sizei tmpl_sz)
{
	int j;
	Recti cutWindow, limRect;
	Recti border;
	IMG_MAT_UCHAR midMat;
	unsigned char *pData;

	UTILS_assert(pdst != NULL);
	UTILS_assert(pdst->width == tmpl_sz.width && pdst->height == tmpl_sz.height);

	limRect.x = 0;	limRect.y = 0;
	limRect.width = src.width;		limRect.height = src.height;

	cutWindow = limit_(window, limRect);
	 if (cutWindow.height <= 0 || cutWindow.width <= 0){	
		 UTILS_assert(0);
	 }

	 border = getBorder(window, cutWindow);
//	 cv::Mat res = in(cutWindow);

	 AllocUcharMat(&midMat, window.width, window.height, 1);

	 if(border.x !=0 || border.y !=0 || border.width != 0 || border.height != 0)
	 {
		 for(j=border.y; j<border.y + cutWindow.height; j++)
		 {
			 memcpy(midMat.data_u8+ j*midMat.step[0]+border.x, 
								 src.data_u8+(j-border.y+cutWindow.y)*src.step[0] + cutWindow.x, 
								 cutWindow.width*sizeof(unsigned char));

			 memset(midMat.data_u8 + j*midMat.step[0], 
						 	*(midMat.data_u8 + j*midMat.step[0]+border.x),
						 	border.x);
			
			 memset(midMat.data_u8 + j*midMat.step[0]+border.x + cutWindow.width, 
						 	*(midMat.data_u8 + j*midMat.step[0]+border.x + cutWindow.width -1),
						 	border.width);
		 }

		 pData = midMat.data_u8 + border.y *midMat.step[0];
		 for(j=0; j<border.y; j++)
		 {
			 memcpy(pData - (j +1)*midMat.step[0],  pData, midMat.step[0]*sizeof(unsigned char));
		 }

		 pData = midMat.data_u8 + (border.y + cutWindow.height)*midMat.step[0]; 
		 for(j=0; j<border.height; j++)
		 {
			 memcpy(pData + j*midMat.step[0],  pData-midMat.step[0], midMat.step[0]*sizeof(unsigned char));
		 }
	 }
	 else
	 {
		 for(j=0; j<cutWindow.height; j++)
		 {
			 memcpy(midMat.data_u8 + j*midMat.step[0], 
								src.data_u8+(j+cutWindow.y)*src.step[0] + cutWindow.x, 
								cutWindow.width*sizeof(unsigned char));
		 }
	 }

//	 ResizeMat(midMat, pdst);
//	 Resize_cubic(midMat, pdst);
	 Resize_cubic2(midMat, pdst);

	 FreeUcharMat(&midMat);
}

void ResizeMat(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst)
{
	int dWidth = pdst->width;
	int dHeight = pdst->height;
	int sWidth = src.width;
	int sHeight = src.height;

	float scaleX = sWidth*1.0f/dWidth;
	float scaleY = sHeight*1.0f/dHeight;
	int ix, iy;
	unsigned char *pSrcData, *pDstData;
	int i, j;

	for(j=0; j<dHeight; j++)
	{
		iy = (int)(scaleY*j);
		pDstData  = pdst->data_u8 + pdst->step[0]*j;
		pSrcData  = src.data_u8 + src.step[0]*iy;

		#pragma UNROLL(4)
		for(i=0; i<dWidth; i++)
		{
			ix = (int)(scaleX*i);
			pDstData[i] = pSrcData[ix];
		}
	}
}

#define clip(minv, maxv, value)		( (value)<minv )?minv:( ( ( value) > maxv)? maxv:(value) )

void ResizeMat_interp(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst)
{
	int dWidth = pdst->width;
	int dHeight = pdst->height;
	int sWidth = src.width;
	int sHeight = src.height;

	float scaleX = sWidth*1.0f/dWidth;
	float scaleY = sHeight*1.0f/dHeight;
	int ix, iy;
	float fx, fy;
	float a_00, a_01, b_00, b_01;
	unsigned char *pSrcData, *pDstData;
	unsigned char C00, C01, C10,C11;
	int i, j, gray;

	for(j=0; j<dHeight-1; j++)
	{
		fy = (scaleY*j);
		iy = (int)fy;
		b_01 = fy - iy;
		b_00 = 1.0 - b_01;
		pDstData  = pdst->data_u8 + pdst->step[0]*j;
		pSrcData  = src.data_u8 + src.step[0]*iy;

		#pragma UNROLL(4)
		for(i=0; i<dWidth; i++)
		{
			fx = (scaleX*i);
			ix = (int)fx;

			a_01 = fx - ix;
			a_00 = 1.0 - a_01;

			C00 = pSrcData[ix];
			C01 = pSrcData[ix+1];
			C10 = pSrcData[ix+ src.step[0]];
			C11 = pSrcData[ix+ src.step[0]+1];

			gray = (C00*a_00 + C01*a_01)*b_00 + (C10*a_00 + C11*a_01)*b_01;

			pDstData[i] = clip(0, 255, gray);
		}
	}
}

static void interpolateCubic( float x, float* coeffs )
{
	const float A = -0.75f;

	coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
	coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
	coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
	coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

#if 0
static void Resize_cubic(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst)
{
	int dWidth = pdst->width;
	int dHeight = pdst->height;
	int sWidth = src.width;
	int sHeight = src.height;

	float scaleX = sWidth*1.0f/dWidth;
	float scaleY = sHeight*1.0f/dHeight;
	int ix, iy;
	float fx, fy;
	float u,v;
	unsigned char *pSrcData, *pDstData;
	float  gray, *coefx, *coefy;
	int i, j, m, n;
	float coefX[1024*4], coefY[768*4];
	int IdxX[1024], IdxY[768];

	for(j=0; j<dHeight; j++){
		fy = (scaleY*j);	iy = (int)fy;	v = fy - iy;
		IdxY[j] = iy;
		interpolateCubic(v, coefY+4*j);
	}
	for(i=0; i<dWidth; i++){
		fx = (scaleX*i);	ix = (int)fx;	u = fx - ix;
		IdxX[i] = ix;
		interpolateCubic(u, coefX+4*i);
	}

	for(j=1; j<dHeight-2; j++)
	{
		iy = IdxY[j];
		coefy = coefY+4*j;
		pDstData  = pdst->data_u8 + pdst->step[0]*j;
		pSrcData  = src.data_u8 + src.step[0]*iy;

#pragma UNROLL(4)
		for(i=1; i<dWidth-2; i++)
		{
			ix = IdxX[i];
			coefx = coefX+4*i;
#if 0
			gray = 0.f;
			for(m=-1; m<=2; m++)
			{
				for(n=-1; n<=2; n++)
				{
					gray += *(pSrcData+m*src.step[0]+n+ix)*coefx[n+1]*coefy[m+1];
				}
			}
#else
			gray = *(pSrcData-src.step[0]-1+ix)*coefx[0]*coefy[0] + *(pSrcData-src.step[0]+ix)*coefx[1]*coefy[0]+
							*(pSrcData-src.step[0]+1+ix)*coefx[2]*coefy[0] + *(pSrcData-src.step[0]+2+ix)*coefx[3]*coefy[0]+
							*(pSrcData-1+ix)*coefx[0]*coefy[1] + *(pSrcData+ix)*coefx[1]*coefy[1]+
							*(pSrcData+1+ix)*coefx[2]*coefy[1] + *(pSrcData+2+ix)*coefx[3]*coefy[1]+
							*(pSrcData+src.step[0]-1+ix)*coefx[0]*coefy[2] + *(pSrcData+src.step[0]+ix)*coefx[1]*coefy[2]+
							*(pSrcData+src.step[0]+1+ix)*coefx[2]*coefy[2] + *(pSrcData+src.step[0]+2+ix)*coefx[3]*coefy[2]+
							*(pSrcData+2*src.step[0]-1+ix)*coefx[0]*coefy[3] + *(pSrcData+2*src.step[0]+ix)*coefx[1]*coefy[3]+
							*(pSrcData+2*src.step[0]+1+ix)*coefx[2]*coefy[3] + *(pSrcData+2*src.step[0]+2+ix)*coefx[3]*coefy[3];

#endif
			pDstData[i] = (unsigned char)clip(0,255, gray);
		}
	}
}
#else
static void Resize_cubic(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst)
{
	int dWidth = pdst->width;
	int dHeight = pdst->height;
	int sWidth = src.width;
	int sHeight = src.height;

	float scaleX = sWidth*1.0f/dWidth;
	float scaleY = sHeight*1.0f/dHeight;
	int ix, iy;
	float fx, fy;
	float u,v;
	unsigned char *pSrcData, *pDstData;
	float  gray, *coefx, *coefy;
	int i, j, m, n;
	float coefX[1024*4], coefY[768*4];
	int IdxX[1024], IdxY[768];
	float32x4_t v_bx0123;
	float32x4_t v_by0000,v_by1111,v_by2222,v_by3333;
	float32x4_t v_bx0123y0000,v_bx0123y1111, v_bx0123y2222,v_bx0123y3333;
	float32x4_t result;

	for(j=0; j<dHeight; j++){
		fy = (scaleY*j);	iy = (int)fy;	v = fy - iy;
		IdxY[j] = iy;
		interpolateCubic(v, coefY+4*j);
	}
	for(i=0; i<dWidth; i++){
		fx = (scaleX*i);	ix = (int)fx;	u = fx - ix;
		IdxX[i] = ix;
		interpolateCubic(u, coefX+4*i);
	}

	for(j=1; j<dHeight-2; j++)
	{
		iy = IdxY[j];
		coefy = coefY+4*j;
		pDstData  = pdst->data_u8 + pdst->step[0]*j;
		pSrcData  = src.data_u8 + src.step[0]*iy;
		v_by0000 = vdupq_n_f32(coefy[0]);
		v_by1111 = vdupq_n_f32(coefy[1]);
		v_by2222 = vdupq_n_f32(coefy[2]);
		v_by3333 = vdupq_n_f32(coefy[3]);

#pragma UNROLL(4)
		for(i=1; i<dWidth-1; i++)
		{
			ix = IdxX[i];
			coefx = coefX+4*i;
			v_bx0123 = vld1q_f32(coefx);
			v_bx0123y0000	 = vmulq_f32(v_bx0123, v_by0000);
			v_bx0123y1111	 = vmulq_f32(v_bx0123, v_by1111);
			v_bx0123y2222	 = vmulq_f32(v_bx0123, v_by2222);
			v_bx0123y3333	 = vmulq_f32(v_bx0123, v_by3333);

			uint8x8_t  A = vld1_u8(pSrcData-src.step[0]-1+ix);
			uint8x8_t  B = vld1_u8(pSrcData-1+ix);
			uint8x8_t  C = vld1_u8(pSrcData+src.step[0]-1+ix);
			uint8x8_t  D = vld1_u8(pSrcData+2*src.step[0]-1+ix);

			uint16x8_t v0_16 = vmovl_u8 (A);			/*将8位扩展为16位*/
			uint16x4_t v0_16_low = vget_low_u16(v0_16); /*读取寄存器的高/低部分到新的寄存器中*/
			uint32x4_t v0_32_low = vmovl_u16(v0_16_low);    /*将16位扩展为32位*/
			float32x4_t A_32f_low = vcvtq_f32_u32(v0_32_low);    /*将int转换为float*/

			uint16x8_t v1_16 = vmovl_u8 (B);			/*将8位扩展为16位*/
			uint16x4_t v1_16_low = vget_low_u16(v1_16); /*读取寄存器的高/低部分到新的寄存器中*/
			uint32x4_t v1_32_low = vmovl_u16(v1_16_low);    /*将16位扩展为32位*/
			float32x4_t B_32f_low = vcvtq_f32_u32(v1_32_low);    /*将int转换为float*/

			uint16x8_t v2_16 = vmovl_u8 (C);			/*将8位扩展为16位 */
			uint16x4_t v2_16_low = vget_low_u16(v2_16); /*读取寄存器的高/低部分到新的寄存器中*/
			uint32x4_t v2_32_low = vmovl_u16(v2_16_low);    /*将16位扩展为32位*/
			float32x4_t C_32f_low = vcvtq_f32_u32(v2_32_low);    /*将int转换为float*/

			uint16x8_t v3_16 = vmovl_u8 (D);			/*将8位扩展为16位*/
			uint16x4_t v3_16_low = vget_low_u16(v3_16); /*读取寄存器的高/低部分到新的寄存器中*/
			uint32x4_t v3_32_low = vmovl_u16(v3_16_low);    /*将16位扩展为32位*/
			float32x4_t D_32f_low = vcvtq_f32_u32(v3_32_low);    /*将int转换为float*/

			result = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_bx0123y0000, A_32f_low),
																										  v_bx0123y1111, B_32f_low),
																										  v_bx0123y2222, C_32f_low),
																										  v_bx0123y3333, D_32f_low);
			gray =  vgetq_lane_f32 (result, 0)+ vgetq_lane_f32 (result, 1)+ vgetq_lane_f32 (result, 2)+ vgetq_lane_f32 (result, 3);
			pDstData[i] = (unsigned char)clip(0,255, gray);
		}
	}
}
#endif

static void Resize_cubic2(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst)
{
	cv::Mat	srcMat, dstMat;
	srcMat = cv::Mat(src.height, src.width, CV_8UC1, src.data_u8);
	dstMat = cv::Mat(pdst->height, pdst->width, CV_8UC1, pdst->data_u8);
	cv::resize( srcMat, dstMat, dstMat.size(), 0, 0, CV_INTER_CUBIC );
}

void GradXY(IMG_MAT_UCHAR src, IMG_MAT_FLOAT *pdst)
{
	int nWidth = pdst->width;
	int nHeight = pdst->height;

	unsigned char *pSrcData;
	float	*pDstData;
	float	dx, dy;
	int i, j;

	assert(src.width == pdst->width && src.height == pdst->height);
	for(j=1; j<nHeight-1; j++)
	{
		pDstData  = pdst->data + pdst->step[0]*j;
		pSrcData  = src.data_u8 + src.step[0]*j;
#pragma UNROLL(4)
		for(i=1; i<nWidth-1; i++)
		{
			dx = (float)pSrcData[i+1]-(float)pSrcData[i-1];
			dy = (float)pSrcData[i+src.step[0]]-(float)pSrcData[i-src.step[0]];
			pDstData[i] = fabs(dx) + fabs(dy);
		}
	}
}

void CutWindowMat(IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst, Recti window)
{
	int j;
	Recti cutWindow, limRect;
	Recti border;
	unsigned char *pData;

	UTILS_assert(pdst != NULL);

	limRect.x = 0;	limRect.y = 0;
	limRect.width = src.width;		limRect.height = src.height;

	cutWindow = limit_(window, limRect);
	 if (cutWindow.height <= 0 || cutWindow.width <= 0){
		 UTILS_assert(0);
	 }

	 border = getBorder(window, cutWindow);

	 if(border.x !=0 || border.y !=0 || border.width != 0 || border.height != 0)
	 {
		 for(j=border.y; j<border.y + cutWindow.height; j++)
		 {
			 memcpy(pdst->data_u8+ j*pdst->step[0]+border.x,
								 src.data_u8+(j-border.y+cutWindow.y)*src.step[0] + cutWindow.x,
								 cutWindow.width*sizeof(unsigned char));

			 memset(pdst->data_u8 + j*pdst->step[0],
						 	*(pdst->data_u8 + j*pdst->step[0]+border.x),
						 	border.x);

			 memset(pdst->data_u8 + j*pdst->step[0]+border.x + cutWindow.width,
						 	*(pdst->data_u8 + j*pdst->step[0]+border.x + cutWindow.width -1),
						 	border.width);
		 }

		 pData = pdst->data_u8 + border.y *pdst->step[0];
		 for(j=0; j<border.y; j++)
		 {
			 memcpy(pData - (j +1)*pdst->step[0],  pData, pdst->step[0]*sizeof(unsigned char));
		 }

		 pData = pdst->data_u8 + (border.y + cutWindow.height)*pdst->step[0];
		 for(j=0; j<border.height; j++)
		 {
			 memcpy(pData + j*pdst->step[0],  pData-pdst->step[0], pdst->step[0]*sizeof(unsigned char));
		 }
	 }
	 else
	 {
		 for(j=0; j<cutWindow.height; j++)
		 {
			 memcpy(pdst->data_u8 + j*pdst->step[0],
								src.data_u8+(j+cutWindow.y)*src.step[0] + cutWindow.x,
								cutWindow.width*sizeof(unsigned char));
		 }
	 }
}

void _IMG_sobel(  IMG_MAT_UCHAR src, IMG_MAT_UCHAR *pdst)
{
	unsigned char *in;       /* Input image data  */
	unsigned char  *out;      /* Output image data */
	short cols,rows;         /* Image dimensions  */
	int H;    /* Horizontal mask result            */
	int V;    /* Vertical mask result                   */
	int O;    /* Sum of horizontal and vertical masks   */
	int i;     /* Input pixel offset						 */
	int o;    /* Output pixel offset.                 */
	int xy;   /* Loop counter.                          */

	int i00, i01, i02;
	int i10,      i12;
	int i20, i21, i22;

	cols = pdst->width;
	rows = pdst->height;
	in = src.data_u8;
	out = pdst->data_u8;

	assert(src.width == pdst->width && src.height == pdst->height);

#pragma UNROLL(4)
	for (xy = 0, i = cols + 1, o = 1;    xy < cols*(rows-2) - 2;    xy++, i++, o++)
	{
		i00=in[i-cols-1]; i01=in[i-cols]; i02=in[i-cols+1];
		i10=in[i     -1];                 i12=in[i     +1];
		i20=in[i+cols-1]; i21=in[i+cols]; i22=in[i+cols+1];

		H = -i00 - 2*i01 -   i02 +   i20 + 2*i21 + i22;
		V = -i00 +   i02 - 2*i10 + 2*i12 -   i20 + i22;
		O = abs(H) + abs(V);

		out[o] = clip(0, 255, O);

	}
}

