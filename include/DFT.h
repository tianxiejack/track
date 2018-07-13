#ifndef _DFT_HEAD_
#define _DFT_HEAD_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>             /* malloc/free declarations */
#include <string.h>             /* memset declaration */
#include <assert.h>
#include <sys/stat.h>
#include "PCTracker.h"
#include "UtcTrack.h"

#define clip(minv, maxv, value)		( (value)<minv )?minv:( ( ( value) > maxv)? maxv:(value) )

#define CR_DXT_FORWARD  0
#define CR_DXT_INVERSE  1
#define CR_DXT_SCALE    2 /* divide result by size of array */
#define CR_DXT_INV_SCALE (CR_DXT_INVERSE + CR_DXT_SCALE)
#define CR_DXT_INVERSE_SCALE	CR_DXT_INV_SCALE
#define CR_DXT_ROWS     4 /* transform each row individually */
#define CR_DXT_MUL_CONJ 8 /* conjugate the second argument of cvMulSpectrums */


enum { CR_DFT_INVERSE=1, CR_DFT_SCALE=2, CR_DFT_ROWS=4, CR_DFT_COMPLEX_OUTPUT=16, CR_DFT_REAL_OUTPUT=32,
				CR_DCT_INVERSE = CR_DFT_INVERSE, CR_DCT_ROWS=CR_DFT_ROWS };

/***************************************************************************************************/
#define CR_PI   3.1415926535897932384626433832795
#define CR_LOG2 0.69314718055994530941723212145818

#define CR_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

/* min & max without jumps */
#define  CR_IMIN(a, b)  ((a) ^ (((a)^(b)) & (((a) < (b)) - 1)))

#define  CR_IMAX(a, b)  ((a) ^ (((a)^(b)) & (((a) > (b)) - 1)))

/**********************************************************************************************/
typedef  unsigned int UINT32;

typedef struct _ComplexCR 
{
	float re;
	float im;
}ComplexCR;

typedef struct _PointICR
{
	int	 x;
	int	 y;
}PointICR;

typedef struct _PointfCR 
{
	float	 x;
	float	 y;
}PointfCR;

typedef struct _csize_t{
	int width;
	int height;
}CSize;

typedef struct  _cpoint_t{
	int x;
	int y;
}CPoint;

#if 0
typedef	 struct _valide_roi_t{
	int	 x;
	int	 y;
	int	 width;
	int	 height;
}VALID_ROI;
#else
typedef	 struct _valide_roi_t{
	int	 start_x;
	int	 start_y;
	int	 valid_w;
	int	 valid_h;
}VALID_ROI;
#endif
enum
{
	CR_TM_SQDIFF        =0,
	CR_TM_SQDIFF_NORMED =1,
	CR_TM_CCORR         =2,
	CR_TM_CCORR_NORMED  =3,
	CR_TM_CCOEFF        =4,
	CR_TM_CCOEFF_NORMED =5
};
//extern unsigned char *_mempace;

void dftCR( IMG_MAT_FLOAT _src0, IMG_MAT_FLOAT _dst, int flags, int nonzero_rows );
void idftCR( IMG_MAT_FLOAT src, IMG_MAT_FLOAT dst, int flags, int nonzero_rows );
void mulSpectrumsCR( IMG_MAT_FLOAT _srcA, IMG_MAT_FLOAT _srcB, IMG_MAT_FLOAT _dst, int flags, bool conjB );

void realCR(IMG_MAT_FLOAT src, IMG_MAT_FLOAT *pReMat);
void imagCR(IMG_MAT_FLOAT src, IMG_MAT_FLOAT *pImMat);
void rearrangeCR(IMG_MAT_FLOAT *src,  IMG_MAT_FLOAT midMat);
void complexMultiCR(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, IMG_MAT_FLOAT dst);
void complexDivisionCR(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, IMG_MAT_FLOAT dst);

int getOptimalDFTSizeCR( int size0 );
void AllocIMGMat(IMG_MAT_FLOAT *pIMG, int nWidth, int nHeight, int channels);
void AllocUcharMat(IMG_MAT_UCHAR *pIMG, int nWidth, int nHeight, int channels);
void *AllocSpaceCR(int nSize);
void FreeSpaceCR(void *ptr);
void FreeIMGMat(IMG_MAT_FLOAT *pIMG);
void FreeUcharMat(IMG_MAT_UCHAR *pIMG);
void MergeUChar2Cplex(unsigned char *srcIMG, int srcWidth, int srcHeight, IMG_MAT_FLOAT *dstIMG);

void MergeFloat2Cplex(float *srcIMG, int srcWidth, int srcHeight, IMG_MAT_FLOAT *dstIMG);
void SplitComplex(IMG_MAT_FLOAT *srcIMG, float *dst0, float *dst1, int dstWidth, int dstHeight);
void AddMat(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, IMG_MAT_FLOAT dst);
void AddK(IMG_MAT_FLOAT src0, float K, IMG_MAT_FLOAT dst);
void ExpMat(IMG_MAT_FLOAT src, IMG_MAT_FLOAT dst, float sigma);
void MinMaxLoc(IMG_MAT_FLOAT src, float* _minval, float* _maxval, PointICR* _minidx, PointICR* _maxidx );
float SubPixelPeak(float left, float center, float right);
int CalTgtStruct(IMG_MAT_UCHAR image, UTC_Rect roiRc, int *SSIM);
void GaussianFilterCR(IMG_MAT_UCHAR *psrc, IMG_MAT_UCHAR *pdst,  UTC_Rect roiRect, UTC_SIZE kelSize);
void BlurCR(IMG_MAT_UCHAR *psrc, IMG_MAT_UCHAR *pdst,  UTC_Rect roiRect, UTC_SIZE kelSize);

void gaussianCorrelation(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, IMG_MAT_FLOAT dst, int size_patch[], bool _hogfeatures, float sigma);
void maxCmp(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1,  IMG_MAT_FLOAT src2, IMG_MAT_FLOAT dst, int size_patch[]);
void createGaussianPeak(IMG_MAT_FLOAT dst, int sizey, int sizex, float padding, float output_sigma_factor);
void MultiplyCR(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, float K, IMG_MAT_FLOAT dst);
void getMinMaxValue(IMG_MAT_FLOAT res, PointfCR *_maxfIdx, float *_peak_value);

void trainCR(IMG_MAT_FLOAT src, IMG_MAT_FLOAT _prob, IMG_MAT_FLOAT *_tmpl, IMG_MAT_FLOAT *_alphaf, float train_interp_factor, float lambda,
					int size_patch[], bool _hogfeatures, float sigma);

void detectCR(IMG_MAT_FLOAT z, IMG_MAT_FLOAT x, IMG_MAT_FLOAT *res,  IMG_MAT_FLOAT _alphaf, int size_patch[], bool _hogfeatures, float sigma);

void matDump_u8(int iStep, IMG_MAT_UCHAR *mat, int offset);
void matDump(int iStep, IMG_MAT_FLOAT *mat, int offset);

void matchTemplateUchar( IMG_MAT_UCHAR _img, IMG_MAT_UCHAR _templ, IMG_MAT_FLOAT _result, int method );
void matchTemplateFloat( IMG_MAT_FLOAT _img, IMG_MAT_FLOAT _templ, IMG_MAT_FLOAT _result, int method );

#include <sys/time.h>
static UInt32 Utils_getCurTimeInUsec()
{
  static int isInit = false;
  static UInt32 initTime=0;
  struct timeval tv;

  if(isInit==false)
  {
      isInit = true;

      if (gettimeofday(&tv, NULL) < 0)
        return 0;

      initTime = (UInt32)(tv.tv_sec * 1000u * 1000 + tv.tv_usec);
  }

  if (gettimeofday(&tv, NULL) < 0)
    return 0;

  return (UInt32)(tv.tv_sec * 1000u * 1000u + tv.tv_usec)-initTime;
}

#endif
