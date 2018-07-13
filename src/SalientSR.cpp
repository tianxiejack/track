#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "DFT.h"
#include  "SalientSR.h"
#include "neon_std.h"

static void resizeSR(IMG_MAT_UCHAR src, IMG_MAT_FLOAT *pdst)
{
	int dWidth = pdst->width;
	int dHeight = pdst->height;
	int sWidth = src.width;
	int sHeight = src.height;

	float scaleX = sWidth*1.0f/dWidth;
	float scaleY = sHeight*1.0f/dHeight;
	int ix, iy;
	unsigned char *pSrcData;
	float	*pDstData;
	int i, j;

	for(j=0; j<dHeight; j++)
	{
		iy = (int)(scaleY*j);
		pDstData  = pdst->data + pdst->step[0]*j;
		pSrcData  = src.data_u8 + src.step[0]*iy;
#pragma UNROLL(4)
		for(i=0; i<dWidth; i++)
		{
			ix = (int)(scaleX*i);
			pDstData[i] = (float)pSrcData[ix];
		}
	}
}

#define clip(minv, maxv, value)		( (value)<minv )?minv:( ( ( value) > maxv)? maxv:(value) )
static void resizeSR_interp(IMG_MAT_UCHAR src, IMG_MAT_FLOAT *pdst)
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
	unsigned char *pSrcData;
	float *pDstData, gray;
	unsigned char C00, C01, C10,C11;
	int i, j;

	for(j=0; j<dHeight-1; j++)
	{
		fy = (scaleY*j);
		iy = (int)fy;
		b_01 = fy - iy;
		b_00 = 1.f - b_01;
		pDstData  = pdst->data + pdst->step[0]*j;
		pSrcData  = src.data_u8 + src.step[0]*iy;

#pragma UNROLL(4)
		for(i=0; i<dWidth; i++)
		{
			fx = (scaleX*i);
			ix = (int)fx;

			a_01 = fx - ix;
			a_00 = 1.f - a_01;

			C00 = pSrcData[ix];
			C01 = pSrcData[ix+1];
			C10 = pSrcData[ix+ src.step[0]];
			C11 = pSrcData[ix+ src.step[0]+1];

			gray = (C00*a_00 + C01*a_01)*b_00 + (C10*a_00 + C11*a_01)*b_01;
			pDstData[i] = gray;
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
static void resizeSR_cubic(IMG_MAT_UCHAR src, IMG_MAT_FLOAT *pdst)
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
	unsigned char *pSrcData;
	float *pDstData, gray, *coefx, *coefy;//coefx[4], coefy[4];
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
		pDstData  = pdst->data + pdst->step[0]*j;
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
			pDstData[i] = gray;
		}
	}
}
#else
static void resizeSR_cubic(IMG_MAT_UCHAR src, IMG_MAT_FLOAT *pdst)
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
	unsigned char *pSrcData;
	float *pDstData, gray, *coefx, *coefy;
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
		pDstData  = pdst->data + pdst->step[0]*j;
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

			uint16x8_t v0_16 = vmovl_u8 (A);			//将8位扩展为16位
			uint16x4_t v0_16_low = vget_low_u16(v0_16); //读取寄存器的高/低部分到新的寄存器中
			uint32x4_t v0_32_low = vmovl_u16(v0_16_low);    //将16位扩展为32位
			float32x4_t A_32f_low = vcvtq_f32_u32(v0_32_low);    //将int转换为float

			uint16x8_t v1_16 = vmovl_u8 (B);			//将8位扩展为16位
			uint16x4_t v1_16_low = vget_low_u16(v1_16); //读取寄存器的高/低部分到新的寄存器中
			uint32x4_t v1_32_low = vmovl_u16(v1_16_low);    //将16位扩展为32位
			float32x4_t B_32f_low = vcvtq_f32_u32(v1_32_low);    //将int转换为float

			uint16x8_t v2_16 = vmovl_u8 (C);			//将8位扩展为16位
			uint16x4_t v2_16_low = vget_low_u16(v2_16); //读取寄存器的高/低部分到新的寄存器中
			uint32x4_t v2_32_low = vmovl_u16(v2_16_low);    //将16位扩展为32位
			float32x4_t C_32f_low = vcvtq_f32_u32(v2_32_low);    //将int转换为float

			uint16x8_t v3_16 = vmovl_u8 (D);			//将8位扩展为16位
			uint16x4_t v3_16_low = vget_low_u16(v3_16); //读取寄存器的高/低部分到新的寄存器中
			uint32x4_t v3_32_low = vmovl_u16(v3_16_low);    //将16位扩展为32位
			float32x4_t D_32f_low = vcvtq_f32_u32(v3_32_low);    //将int转换为float

			result = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_bx0123y0000, A_32f_low),
																										  v_bx0123y1111, B_32f_low),
																										  v_bx0123y2222, C_32f_low),
																										  v_bx0123y3333, D_32f_low);
			gray =  vgetq_lane_f32 (result, 0)+ vgetq_lane_f32 (result, 1)+ vgetq_lane_f32 (result, 2)+ vgetq_lane_f32 (result, 3);
			pDstData[i] = gray;
		}
	}
}
#endif

static void invResizeSR(IMG_MAT_FLOAT src, IMG_MAT_UCHAR *pdst)
{
	int dWidth = pdst->width;
	int dHeight = pdst->height;
	int sWidth = src.width;
	int sHeight = src.height;

	float scaleX = sWidth*1.0f/dWidth;
	float scaleY = sHeight*1.0f/dHeight;
	int ix, iy;
	unsigned char *pDstData;
	float	*pSrcData;
	int i, j;

	for(j=0; j<dHeight; j++)
	{
		iy = (int)(scaleY*j);
		pDstData  = pdst->data_u8 + pdst->step[0]*j;
		pSrcData  = src.data + src.step[0]*iy;
#pragma UNROLL(4)
		for(i=0; i<dWidth; i++)
		{
			ix = (int)(scaleX*i);
			pDstData[i] = (unsigned char)(pSrcData[ix]*255);
		}
	}
}

static void invResizeSR_interp(IMG_MAT_FLOAT src, IMG_MAT_UCHAR *pdst)
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
	float *pSrcData, gray;
	unsigned char *pDstData;
	float C00, C01, C10,C11;
	int i, j;

	for(j=0; j<dHeight-1; j++)
	{
		fy = (scaleY*j);
		iy = (int)fy;
		b_01 = fy - iy;
		b_00 = 1.f - b_01;
		pDstData  = pdst->data_u8 + pdst->step[0]*j;
		pSrcData  = src.data + src.step[0]*iy;

#pragma UNROLL(4)
		for(i=0; i<dWidth; i++)
		{
			fx = (scaleX*i);
			ix = (int)fx;

			a_01 = fx - ix;
			a_00 = 1.f - a_01;

			C00 = pSrcData[ix];
			C01 = pSrcData[ix+1];
			C10 = pSrcData[ix+ src.step[0]];
			C11 = pSrcData[ix+ src.step[0]+1];

			gray = ((C00*a_00 + C01*a_01)*b_00 + (C10*a_00 + C11*a_01)*b_01)*255;
			pDstData[i] = (unsigned char)clip(0, 255, gray);
		}
	}
}

#if 0
static void invResizeSR_cubic(IMG_MAT_FLOAT src, IMG_MAT_UCHAR *pdst)
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
	unsigned char *pDstData;
	float  *pSrcData, gray, *coefx, *coefy;//coefx[4], coefy[4];
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
// 		fy = (scaleY*j);
// 		iy = (int)fy;
// 		v = fy - iy;
// 		interpolateCubic(v, coefy);
		iy = IdxY[j];
		coefy = coefY+4*j;
		pDstData  = pdst->data_u8 + pdst->step[0]*j;
		pSrcData  = src.data + src.step[0]*iy;

#pragma UNROLL(4)
		for(i=1; i<dWidth-2; i++)
		{
// 			fx = (scaleX*i);
// 			ix = (int)fx;
// 			u = fx - ix;
// 			interpolateCubic(u, coefx);
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
			gray*=255;
			pDstData[i] = (unsigned char)clip(0, 255, gray);
		}
	}
}
#else
static void invResizeSR_cubic(IMG_MAT_FLOAT src, IMG_MAT_UCHAR *pdst)
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
	unsigned char *pDstData;
	float  *pSrcData, gray, *coefx, *coefy;
	int i, j, m, n;
	float coefX[1024*4], coefY[768*4];
	int IdxX[1024], IdxY[768];
	float32x4_t v_bx0123;
	float32x4_t v_by0000,v_by1111,v_by2222,v_by3333;
	float32x4_t v_bx0123y0000,v_bx0123y1111, v_bx0123y2222,v_bx0123y3333;
	float32x4_t result;
	float *S0,*S1,*S2,*S3;

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
		pSrcData  = src.data + src.step[0]*iy;
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

			S0 = pSrcData-src.step[0]-1+ix;
			S1 = pSrcData-1+ix;
			S2 = pSrcData+src.step[0]-1+ix;
			S3 = pSrcData+2*src.step[0]-1+ix;
			result = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_bx0123y0000, vld1q_f32(S0)),
																										  v_bx0123y1111, vld1q_f32(S1)),
																										  v_bx0123y2222, vld1q_f32(S2)),
																										  v_bx0123y3333, vld1q_f32(S3));
			gray =  vgetq_lane_f32 (result, 0)+ vgetq_lane_f32 (result, 1)+ vgetq_lane_f32 (result, 2)+ vgetq_lane_f32 (result, 3);
			gray*=255;
			pDstData[i] = (unsigned char)clip(0, 255, gray);
		}
	}
}
#endif

static void blurSubSR(IMG_MAT_FLOAT *psrc, IMG_MAT_FLOAT *pdst,  UTC_SIZE kelSize)
{
	int i, j, x, y;
	int width = psrc->width;
	int height = psrc->height;
	int kelArean = kelSize.width*kelSize.height;
	float *psrcdata, *pdstdata, *pmiddata;
	float sum;
	assert(psrc != NULL && pdst != NULL);
	assert(psrc->width == pdst->width && psrc->height == pdst->height);

	for(y=kelSize.height/2; y<height-kelSize.height/2; y++)
	{
		for(x=kelSize.width/2; x<width - kelSize.width/2; x++)
		{
			psrcdata = psrc->data + y*psrc->width + x;
			pdstdata = pdst->data + y*pdst->width + x;
			sum = 0.f;
			for(j=0; j<kelSize.height; j++)
			{
				pmiddata = psrcdata + (j-kelSize.height/2)*psrc->width - kelSize.width/2;
				for(i=0; i<kelSize.width; i++)
				{
					sum += pmiddata[i];
				}
			}
			pdstdata[0] = psrcdata[0] - sum/kelArean;
		}
	}
}

static const int SMALL_GAUSSIAN_SIZE = 7;
static const float small_gaussian_tab[4][7] =
{
	{1.f},
	{0.25f, 0.5f, 0.25f},
	{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
	{0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
};

static void GaussianBlurSR(IMG_MAT_FLOAT *psrc, IMG_MAT_FLOAT *pdst,  UTC_SIZE kelSize, float *pMin, float *pMax)
{
	float  maxval = 0;
	float minval = 3.402823466e+38F;
	int i, j, x, y;
	int width = psrc->width;
	int height = psrc->height;
	int kelArean = kelSize.width*kelSize.height;
	float *psrcdata, *pdstdata,*pmiddata;
	float sum = 0.f;
	float gaussian2DTab[7][7] ={0,};
	int Idx = kelSize.width/2;

	assert(kelSize.width == kelSize.height && (kelSize.width == 3 || kelSize.width  == 5 || kelSize.width == 7));
	assert(psrc != NULL && pdst != NULL);
	assert(psrc->width == pdst->width && psrc->height == pdst->height);

	for(j=0; j<kelSize.height; j++)
	{
		for(i=0; i<kelSize.width; i++)
		{
			gaussian2DTab[j][i] = small_gaussian_tab[Idx][i]*small_gaussian_tab[Idx][j];
			sum += gaussian2DTab[j][i];
		}
	}
	for(j=0; j<kelSize.height; j++)
	{
		for(i=0; i<kelSize.width; i++)
		{
			gaussian2DTab[j][i] =gaussian2DTab[j][i]/sum;
		}
	}

	for(y=kelSize.height/2; y<height-kelSize.height/2; y++)
	{
		for(x=kelSize.width/2; x<width - kelSize.width/2; x++)
		{
			psrcdata = psrc->data + y*psrc->width + x;
			pdstdata = pdst->data + y*pdst->width + x;
			sum = 0.f;
			for(j=0; j<kelSize.height; j++)
			{
				pmiddata = psrcdata + (j-kelSize.height/2)*psrc->width - kelSize.width/2;
				for(i=0; i<kelSize.width; i++)
				{
					sum += pmiddata[i]*gaussian2DTab[j][i];
				}
			}
			pdstdata[0] = sum;
			if( maxval < sum) maxval = sum;
			if( minval > sum) minval = sum;
		}
	}
	if(pMin != NULL) *pMin = minval;
	if(pMax != NULL) *pMax = maxval;
}

static void logSR(IMG_MAT_FLOAT *psrc, IMG_MAT_FLOAT *pdst)
{
	int width = psrc->width;
	int height = psrc->height;
	int k;
	int size = width*height;
	float *psrcdata, *pdstdata;

	assert(psrc != NULL && pdst != NULL);
	assert(psrc->width == pdst->width && psrc->height == pdst->height);

	psrcdata = psrc->data;
	pdstdata = pdst->data;
	#pragma UNROLL(4)
	for(k=0; k<size; k++)
	{
		*pdstdata++ = (float)log(*psrcdata++);
	}
}

static void expSR(IMG_MAT_FLOAT *psrc, IMG_MAT_FLOAT *pdst)
{
	int width = psrc->width;
	int height = psrc->height;
	int k;
	int size = width*height;
	float *psrcdata, *pdstdata;
	assert(psrc != NULL && pdst != NULL);
	assert(psrc->width == pdst->width && psrc->height == pdst->height);

	psrcdata = psrc->data;
	pdstdata = pdst->data;
#pragma UNROLL(4)
	for(k=0; k<size; k++)
	{
		*pdstdata++ = (float)exp(*psrcdata++);
	}
}

static void spectrEnergy(IMG_MAT_FLOAT *psrc, IMG_MAT_FLOAT *pdst)
{
	int width = psrc->width;
	int height = psrc->height;
	int k;
	int size = width*height;
	ComplexCR *psrcData;
	float *pdstdata;

	assert(psrc->width == pdst->width && psrc->height == pdst->height);
	assert(psrc->channels == 2 && pdst->channels == 1);

	psrcData = (ComplexCR *)psrc->data;
	pdstdata = pdst->data;
#pragma UNROLL(4)
	for(k=0; k<size; k++)
	{
		pdstdata[k]= psrcData[k].re*psrcData[k].re + psrcData[k].im*psrcData[k].im;
	}
}

static void cartToPolarSR(IMG_MAT_FLOAT src, IMG_MAT_FLOAT magnitude, IMG_MAT_FLOAT angle)
{
	int width = src.width;
	int height = src.height;
	int k, size = width*height;
	float *pMagData, *pAngData;
	ComplexCR *psrcData;

	assert(src.width == magnitude.width && src.height == magnitude.height);
	assert(src.width == angle.width && src.height == angle.height);
	assert(src.channels == 2 && magnitude.channels == 1 && angle.channels == 1);

	psrcData = (ComplexCR *)src.data;
	pMagData = (float*)magnitude.data;
	pAngData = (float*)angle.data;
	#pragma UNROLL(4)
	for(k=0; k<size; k++)
	{
		pAngData[k] = (float)atan2(psrcData[k].im, psrcData[k].re);
		pMagData[k] = (float)sqrt(psrcData[k].re * psrcData[k].re + psrcData[k].im * psrcData[k].im);
	}
}

static void polarToCartSR(IMG_MAT_FLOAT dst, IMG_MAT_FLOAT magnitude, IMG_MAT_FLOAT angle)
{
	int width = dst.width;
	int height = dst.height;
	int k, size = width*height;
	float *pMagData, *pAngData;
	ComplexCR *pdstData;

	assert(dst.width == magnitude.width && dst.height == magnitude.height);
	assert(dst.width == angle.width && dst.height == angle.height);
	assert(dst.channels == 2 && magnitude.channels == 1 && angle.channels == 1);

	pdstData = (ComplexCR *)dst.data;
	pMagData = (float*)magnitude.data;
	pAngData = (float*)angle.data;
#pragma UNROLL(4)
	for(k=0; k<size; k++)
	{
		pdstData[k].re = (float)(pMagData[k] * cos(pAngData[k]));
		pdstData[k].im = (float)(pMagData[k] * sin(pAngData[k]));
	}
}

static void normalizeSR(IMG_MAT_FLOAT *psrc , IMG_MAT_FLOAT *pdst, float minval, float maxval)
{
	int width = psrc->width;
	int height = psrc->height;
	int k;
	int size = width*height;
	float *psrcdata, *pdstdata, range;
	assert(psrc != NULL && pdst != NULL);
	assert(psrc->width == pdst->width && psrc->height == pdst->height);

	psrcdata = psrc->data;
	pdstdata = pdst->data;
	range = 1/(maxval - minval);
#pragma UNROLL(4)
	for(k=0; k<size; k++)
	{
		*pdstdata++ = (*psrcdata++ - minval)*range;
	}
}

void SalientSR(IMG_MAT src, IMG_MAT salientMat, UTC_Rect acqWin, UTC_Rect srROI)
{
	UTC_SIZE	scaleSZ, kelSize;
	float  maxval, minval;
	IMG_MAT_FLOAT	resizeMat, cmplxMat, magMat, angMat, blurMat, spectMat, gaussMat;
	scaleSZ.width = 64;	scaleSZ.height = 64;
	kelSize.width = 5;	kelSize.height = 5;
	AllocIMGMat(&resizeMat, scaleSZ.width, scaleSZ.height, 1);
	AllocIMGMat(&cmplxMat, scaleSZ.width, scaleSZ.height, 2);
	AllocIMGMat(&magMat, scaleSZ.width, scaleSZ.height, 1);
	AllocIMGMat(&angMat, scaleSZ.width, scaleSZ.height, 1);
	AllocIMGMat(&blurMat, scaleSZ.width, scaleSZ.height, 1);
	AllocIMGMat(&spectMat, scaleSZ.width, scaleSZ.height, 1);
	AllocIMGMat(&gaussMat, scaleSZ.width, scaleSZ.height, 1);

	resizeSR_cubic(src, &resizeMat);
	MergeFloat2Cplex(resizeMat.data, scaleSZ.width, scaleSZ.height, &cmplxMat);
	dftCR(cmplxMat, cmplxMat, 0, 0);
	cartToPolarSR(cmplxMat, magMat, angMat);
	logSR(&magMat, &magMat);
	blurSubSR(&magMat, &blurMat, kelSize);
	expSR(&blurMat, &blurMat);
	polarToCartSR(cmplxMat, blurMat, angMat);
	idftCR(cmplxMat, cmplxMat, CR_DFT_INVERSE | CR_DFT_SCALE, 0);
	spectrEnergy(&cmplxMat, &spectMat);
	GaussianBlurSR(&spectMat, &gaussMat, kelSize, &minval, &maxval);
	normalizeSR(&gaussMat, &gaussMat, minval, maxval);
	invResizeSR_cubic(gaussMat, &salientMat);
	{
		int j;
		memset(salientMat.data_u8, 0x00, salientMat.width*8);
		memset((salientMat.data_u8+(salientMat.height-8)*salientMat.step[0]), 0x00, salientMat.width*8);
		for(j=0; j<salientMat.height; j++){
			memset(salientMat.data_u8+j*salientMat.step[0], 0x00, 8);
			memset(salientMat.data_u8+(j+1)*salientMat.step[0]-8, 0x00, 8);
		}
	}

	FreeIMGMat(&resizeMat);
	FreeIMGMat(&cmplxMat);
	FreeIMGMat(&magMat);
	FreeIMGMat(&angMat);
	FreeIMGMat(&blurMat);
	FreeIMGMat(&spectMat);
	FreeIMGMat(&gaussMat);
}
