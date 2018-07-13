#include "DFT.h"
#include "PCTracker.h"
#include <float.h>

static void convert_F2F(IMG_MAT_FLOAT	*src,	IMG_MAT_FLOAT	*dst)
{
	int i, j;
	float *psrc,*pdst;

	assert(src != NULL && dst != NULL);

	if(dst->width == src->width )
	{
		psrc =  (float*)src->data ;
		pdst =  (float*)dst->data ;
		memcpy(pdst, psrc, src->width*src->height*sizeof(float));
	}
	else if( dst->width > src->width )
	{
#pragma UNROLL(4)
		for(j=0; j<src->height; j++)
		{
			psrc = (float*)src->data + src->step[0]*j;
			pdst = (float*)dst->data + dst->step[0]*j;
			memcpy(pdst, psrc, sizeof(float)*src->width);
			memset(pdst+src->width, 0x00, sizeof(float)*(dst->width-src->width));
		}
	}
}

static void convert_F2F_ROI(IMG_MAT_FLOAT	*src,	IMG_MAT_FLOAT	*dst,	VALID_ROI	*src_roi, VALID_ROI	*dst_roi)
{
	int i, j;
	float *psrc,	*pdst;
	assert(src != NULL && dst != NULL && src_roi != NULL && dst_roi != NULL);

#pragma UNROLL(4)
	for(j=0; j<src_roi->valid_h; j++)
	{
		psrc = (float *)(src->data+(src_roi->start_y+j)*src->step[0] + src_roi->start_x);
		pdst = (float*)(dst->data+(dst_roi->start_y+j)*dst->step[0] + dst_roi->start_x);
		memcpy(pdst, psrc, src_roi->valid_w*sizeof(float));
	}
}

static void crossCorr( const IMG_MAT_FLOAT img, const IMG_MAT_FLOAT _templ, IMG_MAT_FLOAT corr, 
	CSize corrsize, int ctype,	CPoint anchor, double delta, int borderType )
{
	const double blockScale = 4.5;
	const int minBlockSize = 256;
	IMG_MAT_FLOAT templ = _templ;
	IMG_MAT_FLOAT	dftTempl,	dftImg;
	int i, j , k;
	CSize blocksize, dftsize;
	int tileCountX , tileCountY, tileCount;
	CPoint roiofs;//(0,0);
	IMG_MAT_FLOAT img0;

	assert( corrsize.height <= img.height + templ.height - 1 && corrsize.width <= img.width + templ.width - 1 );

	blocksize.width = floor(templ.width*blockScale);
	blocksize.width = MAX( blocksize.width, minBlockSize - templ.width + 1 );
	blocksize.width = MIN( blocksize.width, corr.width );
	blocksize.height = floor(templ.height*blockScale);
	blocksize.height = MAX( blocksize.height, minBlockSize - templ.height + 1 );
	blocksize.height = MIN( blocksize.height, corr.height );

	dftsize.width = MAX(getOptimalDFTSizeCR(blocksize.width + templ.width - 1), 2);
	dftsize.height = getOptimalDFTSizeCR(blocksize.height + templ.height - 1);

	assert( dftsize.width > 0 && dftsize.height > 0 );

	// recompute block size
	blocksize.width = dftsize.width - templ.width + 1;
	blocksize.width = MIN( blocksize.width, corr.width );
	blocksize.height = dftsize.height - templ.height + 1;
	blocksize.height = MIN( blocksize.height, corr.height );

	AllocIMGMat(&dftTempl, dftsize.width, dftsize.height, 1);
	AllocIMGMat(&dftImg, dftsize.width, dftsize.height, 1);
	// compute DFT of each template plane
	for( k = 0; k < 1; k++ )
	{
		IMG_MAT_FLOAT dst = dftTempl;
		convert_F2F(&templ, &dst);
		dftCR(dst, dst, 0, templ.height);
	}

	tileCountX = (corr.width + blocksize.width - 1)/blocksize.width;
	tileCountY = (corr.height + blocksize.height - 1)/blocksize.height;
	tileCount = tileCountX * tileCountY;
	img0 = img;
	roiofs.x = 0; roiofs.y = 0;

	// calculate correlation by blocks
	for( i = 0; i < tileCount; i++ )
	{
		int x = (i%tileCountX)*blocksize.width;
		int y = (i/tileCountX)*blocksize.height;
		int x0, y0, x1, y1, x2, y2;
		IMG_MAT_FLOAT src0;//(img0, Range(y1, y2), Range(x1, x2));
		IMG_MAT_FLOAT dst;//(dftImg, Rect(0, 0, dsz.width, dsz.height));
		IMG_MAT_FLOAT dst1;//(dftImg, Rect(x1-x0, y1-y0, x2-x1, y2-y1));
		IMG_MAT_FLOAT cdst;//(corr, Rect(x, y, bsz.width, bsz.height));
		IMG_MAT_FLOAT src;//dftImg(Rect(0, 0, bsz.width, bsz.height));
		VALID_ROI	src0_roi,	dst_roi, dst1_roi, cdst_roi, src_roi;

		CSize bsz,	dsz;
		bsz.width = MIN(blocksize.width, corr.width - x);
		bsz.height = MIN(blocksize.height, corr.height - y);

		dsz.width = bsz.width + templ.width - 1;
		dsz.height = bsz.height + templ.height - 1;
		x0 = x - anchor.x + roiofs.x, y0 = y - anchor.y + roiofs.y;
		x1 = MAX(0, x0), y1 = MAX(0, y0);
		x2 = MIN(img0.width, x0 + dsz.width);
		y2 = MIN(img0.height, y0 + dsz.height);

		src0 = img0;	
		src0_roi.start_x = x1;	src0_roi.start_y = y1;
		src0_roi.valid_w = x2-x1;	src0_roi.valid_h = y2-y1;
		dst = dftImg;
		dst_roi.start_x = 0;					dst_roi.start_y = 0;
		dst_roi.valid_w = dsz.width;	dst_roi.valid_h = dsz.height;
		dst1 = dftImg;
		dst1_roi.start_x = x1-x0;		dst1_roi.start_y = y1-y0;
		dst1_roi.valid_w = x2-x1;	dst1_roi.valid_h = y2-y1;
		cdst = corr;
		cdst_roi.start_x = x;						cdst_roi.start_y = y;
		cdst_roi.valid_w = bsz.width;	cdst_roi.valid_h = bsz.height;
		{
			float *pImg;
			pImg = (float*)dftImg.data;
			memset(pImg, 0x00, dftsize.width*dftsize.height*sizeof(float));

			convert_F2F_ROI(&src0, &dst1, &src0_roi, &dst1_roi);
			dftCR(dftImg, dftImg, 0, dsz.height);

			mulSpectrumsCR(dftImg, dftTempl, dftImg, 0, true);
			idftCR( dftImg, dftImg, CR_DFT_INVERSE + CR_DFT_SCALE, bsz.height );

			src = dftImg;
			src_roi.start_x = 0;					src_roi.start_y = 0;
			src_roi.valid_w = bsz.width;	src_roi.valid_h = bsz.height;
			convert_F2F_ROI(&src, &cdst, &src_roi, &cdst_roi);
		}
	}

	FreeIMGMat(&dftTempl);
	FreeIMGMat(&dftImg);
}

#if 0
static void Integral(IMG_MAT_FLOAT _img, IMG_MAT_DOUBLE _sum, IMG_MAT_DOUBLE _sqsum, int cn)
{
	int x, y, k;
	int nWidth, nHeight;
	int imgStep, sumStep, sqsumStep;
	float *pImg;
	double *pSum, *pSqSum;
	nWidth = _img.width;
	nHeight = _img.height;
	pImg = (float *)_img.data;
	pSum = (double*)_sum.data_db;
	pSqSum = (double *)_sqsum.data_db;
	imgStep = _img.step[0];
	sumStep = _sum.step[0];
	sqsumStep = _sqsum.step[0];

	assert(cn == 1);

	memset( pSum, 0, (nWidth+cn)*sizeof(pSum[0]));
	pSum += sumStep + cn;

	memset( pSqSum, 0, (nWidth+cn)*sizeof(pSqSum[0]));
	pSqSum += sqsumStep + cn;

	for( y = 0; y < nHeight; y++, pImg += imgStep,	pSum += sumStep, pSqSum += sqsumStep )
	{
		double s = pSum[-cn] = 0;
		double sq = pSqSum[-cn] = 0;
#pragma UNROLL(4)
		for( x = 0; x < nWidth; x += cn )
		{
			float it = pImg[x];
			double t, tq;
			s += it;
			sq += (double)it*it;
			t = pSum[x - sumStep] + s;
			tq = pSqSum[x - sqsumStep] + sq;
			pSum[x] = t;
			pSqSum[x] = tq;
		}
	}
}

static void meanStdDev(IMG_MAT_FLOAT _img, double *mean, double *stddev, int cn)
{
	int i, j, k;
	int nWidth, nHeight, len;
	int imgStep;
	float *pImg;
	double scale;

	nWidth = _img.width;
	nHeight = _img.height;
	pImg = _img.data;
	imgStep = _img.step[0];

	k = cn % 4;
	len = nWidth*nHeight;

	assert(cn<=3);

	if( k == 1 )
	{
		double s0, sq0;
		mean[0] = 0.0;
		stddev[0] = 0.0;
		s0 = mean[0];
		sq0 = stddev[0];
#pragma UNROLL(4)
		for( i = 0; i < len; i++, pImg += cn )
		{
			float	v = pImg[0];
			s0 += v; sq0 += (double)v*v;
		}
		mean[0] = s0;
		stddev[0] = sq0;
	}
	else if( k == 2 )
	{
		double s0, s1, sq0, sq1;
		mean[0] = mean[1] = 0.0;
		stddev[0] = stddev[1] = 0.0;
		s0 = mean[0], s1 = mean[1];
		sq0 = stddev[0], sq1 = stddev[1];
#pragma UNROLL(4)
		for( i = 0; i < len; i++, pImg += cn )
		{
			float v0 = pImg[0], v1 = pImg[1];
			s0 += v0; sq0 += (double)v0*v0;
			s1 += v1; sq1 += (double)v1*v1;
		}
		mean[0] = s0; mean[1] = s1;
		stddev[0] = sq0; stddev[1] = sq1;
	}
	else if( k == 3 )
	{
		double s0, s1, s2, sq0, sq1, sq2;
		mean[0] = mean[1] = mean[2] =0.0;
		stddev[0] = stddev[1] = stddev[2] = 0.0;
		s0 = mean[0], s1 = mean[1], s2 = mean[2];
		sq0 = stddev[0], sq1 = stddev[1], sq2 = stddev[2];
#pragma UNROLL(4)
		for( i = 0; i < len; i++, pImg += cn )
		{
			float v0 = pImg[0], v1 = pImg[1], v2 = pImg[2];
			s0 += v0; sq0 += (double)v0*v0;
			s1 += v1; sq1 += (double)v1*v1;
			s2 += v2; sq2 += (double)v2*v2;
		}
		mean[0] = s0; mean[1] = s1; mean[2] = s2;
		stddev[0] = sq0; stddev[1] = sq1; stddev[2] = sq2;
	}
	scale = len ? 1./len : 0.;
	for( k = 0; k < cn; k++ )
	{
		mean[k] *= scale;
		stddev[k] = sqrt(MAX(stddev[k]*scale - mean[k]*mean[k], 0.));
	}
}
#else
static void Integral(IMG_MAT_FLOAT _img, IMG_MAT_FLOAT _sum, IMG_MAT_FLOAT _sqsum, int cn)
{
	int x, y, k;
	int nWidth, nHeight;
	int imgStep, sumStep, sqsumStep;
	float *pImg;
	float *pSum, *pSqSum;
	nWidth = _img.width;
	nHeight = _img.height;
	pImg = (float *)_img.data;
	pSum = (float*)_sum.data;
	pSqSum = (float *)_sqsum.data;
	imgStep = _img.step[0];
	sumStep = _sum.step[0];
	sqsumStep = _sqsum.step[0];

	assert(cn == 1);

	memset( pSum, 0, (nWidth+cn)*sizeof(pSum[0]));
	pSum += sumStep + cn;

	memset( pSqSum, 0, (nWidth+cn)*sizeof(pSqSum[0]));
	pSqSum += sqsumStep + cn;

	for( y = 0; y < nHeight; y++, pImg += imgStep,	pSum += sumStep, pSqSum += sqsumStep )
	{
		float s = pSum[-cn] = 0;
		float sq = pSqSum[-cn] = 0;
#pragma UNROLL(4)
		for( x = 0; x < nWidth; x += cn )
		{
			float it = pImg[x];
			float t, tq;
			s += it;
			sq += (float)it*it;
			t = pSum[x - sumStep] + s;
			tq = pSqSum[x - sqsumStep] + sq;
			pSum[x] = t;
			pSqSum[x] = tq;
		}
	}
}

static void meanStdDev(IMG_MAT_FLOAT _img, float *mean, float *stddev, int cn)
{
	int i, j, k;
	int nWidth, nHeight, len;
	int imgStep;
	float *pImg;
	double scale;

	nWidth = _img.width;
	nHeight = _img.height;
	pImg = _img.data;
	imgStep = _img.step[0];

	k = cn % 4;
	len = nWidth*nHeight;

	assert(cn<=3);

	if( k == 1 )
	{
		float s0, sq0;
		mean[0] = 0.0;
		stddev[0] = 0.0;
		s0 = mean[0];
		sq0 = stddev[0];
#pragma UNROLL(4)
		for( i = 0; i < len; i++, pImg += cn )
		{
			float	v = pImg[0];
			s0 += v; sq0 += (float)v*v;
		}
		mean[0] = s0;
		stddev[0] = sq0;
	}
	else if( k == 2 )
	{
		float s0, s1, sq0, sq1;
		mean[0] = mean[1] = 0.0;
		stddev[0] = stddev[1] = 0.0;
		s0 = mean[0], s1 = mean[1];
		sq0 = stddev[0], sq1 = stddev[1];
#pragma UNROLL(4)
		for( i = 0; i < len; i++, pImg += cn )
		{
			float v0 = pImg[0], v1 = pImg[1];
			s0 += v0; sq0 += (float)v0*v0;
			s1 += v1; sq1 += (float)v1*v1;
		}
		mean[0] = s0; mean[1] = s1;
		stddev[0] = sq0; stddev[1] = sq1;
	}
	else if( k == 3 )
	{
		float s0, s1, s2, sq0, sq1, sq2;
		mean[0] = mean[1] = mean[2] =0.0;
		stddev[0] = stddev[1] = stddev[2] = 0.0;
		s0 = mean[0], s1 = mean[1], s2 = mean[2];
		sq0 = stddev[0], sq1 = stddev[1], sq2 = stddev[2];
#pragma UNROLL(4)
		for( i = 0; i < len; i++, pImg += cn )
		{
			float v0 = pImg[0], v1 = pImg[1], v2 = pImg[2];
			s0 += v0; sq0 += (float)v0*v0;
			s1 += v1; sq1 += (float)v1*v1;
			s2 += v2; sq2 += (float)v2*v2;
		}
		mean[0] = s0; mean[1] = s1; mean[2] = s2;
		stddev[0] = sq0; stddev[1] = sq1; stddev[2] = sq2;
	}
	scale = len ? 1./len : 0.;
	for( k = 0; k < cn; k++ )
	{
		mean[k] *= scale;
		stddev[k] = (float)sqrt(MAX(stddev[k]*scale - mean[k]*mean[k], 0.));
	}
}
#endif

void matchTemplateFloat( IMG_MAT_FLOAT _img, IMG_MAT_FLOAT _templ, IMG_MAT_FLOAT _result, int method )
{
	CSize corrSize;
	CPoint anchor;
	int i, j, k;
	IMG_MAT_FLOAT sum, sqsum;
	int numType;
	bool isNormed;

	assert( CR_TM_SQDIFF <= method && method <= CR_TM_CCOEFF_NORMED );

	numType = method == CR_TM_CCORR || method == CR_TM_CCORR_NORMED ? 0 :
		method == CR_TM_CCOEFF || method == CR_TM_CCOEFF_NORMED ? 1 : 2;
	isNormed = method == CR_TM_CCORR_NORMED ||
		method == CR_TM_SQDIFF_NORMED ||
		method == CR_TM_CCOEFF_NORMED;

	corrSize.width = _img.width - _templ.width + 1 ;
	corrSize.height = _img.height - _templ.height + 1;
	anchor.x = 0; anchor.y = 0;
	crossCorr( _img, _templ, _result, corrSize, 0, anchor, 0, 0);

	if( method == CR_TM_CCORR )
		return;

	{
		double invArea = (double)1./(_templ.height * _templ.width);
		float templMean[4], templSdv[4];
		float *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
		double templNorm = 0, templSum2 = 0;
		int cn = 1;
		float* p0, *p1, *p2, *p3;
		int sumstep, sqstep;

		AllocIMGMat(&sum, _img.width+1, _img.height+1, cn);
		AllocIMGMat(&sqsum, _img.width+1, _img.height+1, cn);

		Integral(_img, sum, sqsum, cn);
		meanStdDev( _templ, templMean, templSdv ,cn);
		templNorm = (templSdv[0]*templSdv[0]);

		if( templNorm < DBL_EPSILON  && method == CR_TM_CCOEFF_NORMED )
		{
			float *pRslt = _result.data;
			for(i=0 ; i<_result.width*_result.height; i++){
				*pRslt++= 1.0;
			}
			return;
		}
		templSum2 = templNorm +(templMean[0]*templMean[0]); 

		if( numType != 1 )
		{
			templMean[0] = templMean[1] = templMean[2] = templMean[3] = 0.f;
			templNorm = templSum2;
		}

		templSum2 /= invArea;
		templNorm = sqrt(templNorm);
		templNorm /= sqrt(invArea); // care of accuracy here

		q0 = (float*)sqsum.data;
		q1 = q0 + _templ.width*cn;
		q2 = (float*)(sqsum.data + _templ.height*sqsum.step[0]);
		q3 = q2 + _templ.width*cn;

		p0 = (float*)sum.data;
		p1 = p0 + _templ.width*cn;
		p2 = (float*)(sum.data + _templ.height*sum.step[0]);
		p3 = p2 + _templ.width*cn;

		sumstep = sum.data ? (int)(sum.step[0] ) : 0;
		sqstep = sqsum.data ? (int)(sqsum.step[0]) : 0;

		for( i = 0; i < _result.height; i++ )
		{
			float* rrow = (float*)(_result.data + i*_result.step[0]);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			for( j = 0; j < _result.width; j++, idx += cn, idx2 += cn )
			{
				double num = rrow[j], t;
				double wndMean2 = 0, wndSum2 = 0;

				if( numType == 1 )
				{
					for( k = 0; k < cn; k++ )
					{
						t = p0[idx+k] - p1[idx+k] - p2[idx+k] + p3[idx+k];
						wndMean2 += (t*t);
						num -= t*templMean[k];
					}

					wndMean2 *= invArea;
				}

				if( isNormed || numType == 2 )
				{
					for( k = 0; k < cn; k++ )
					{
						t = q0[idx2+k] - q1[idx2+k] - q2[idx2+k] + q3[idx2+k];
						wndSum2 += t;
					}

					if( numType == 2 )
					{
						num = wndSum2 - 2*num + templSum2;
						num = MAX(num, 0.);
					}
				}

				if( isNormed )
				{
					t = sqrt(MAX(wndSum2 - wndMean2,0))*templNorm;
					if( fabs(num) < t )
						num /= t;
					else if( fabs(num) < t*1.125 )
						num = num > 0 ? 1 : -1;
					else
						num = method != CR_TM_SQDIFF_NORMED ? 0 : 1;
				}

				rrow[j] = (float)num;
			}
		}
	}
	
	FreeIMGMat(&sum);
	FreeIMGMat(&sqsum);
}
