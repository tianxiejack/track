#include "PCTracker.h"
#include "DFT.h"
#include "malloc_align.h"

#if 0
void AllocIMGMat(IMG_MAT_FLOAT *pIMG, int nWidth, int nHeight, int channels)
{
	if(/*pIMG->data == NULL*/1){
//		pIMG->data  = (float*)MallocAlign(nWidth*nHeight*channels*sizeof(float));
		UTILS_assert(pOrigAddress != 0);
		pIMG->data = (float*)(pOrigAddress+PointPos);
		PointPos += ((nWidth*nHeight*channels*sizeof(float)+15)&(~15));
		UTILS_assert(pIMG->data != NULL);
		UTILS_assert(PointPos < SpaceNum);
		memset(pIMG->data, 0x00, nWidth*nHeight*channels*sizeof(float));
	}
	pIMG->dtype = 1;
	pIMG->size = nWidth*nHeight*channels*sizeof(float);
	pIMG->width = nWidth;
	pIMG->height = nHeight;
	pIMG->channels = channels;
	pIMG->step[0] = nWidth*channels;
}

void AllocUcharMat(IMG_MAT_UCHAR *pIMG, int nWidth, int nHeight, int channels)
{
	if(/*pIMG->data == NULL*/1){
		//		pIMG->data  = (float*)MallocAlign(nWidth*nHeight*channels*sizeof(float));
		UTILS_assert(pOrigAddress != 0);
		pIMG->data_u8 = (unsigned char*)(pOrigAddress+PointPos);
		PointPos += ((nWidth*nHeight*channels*sizeof(unsigned char)+15)&(~15));
		UTILS_assert(pIMG->data != NULL);
		UTILS_assert(PointPos < SpaceNum);
		Vps_printf("ttt 26 0x%08x %d %d %d %d\n",
				pIMG->data_u8, nWidth,nHeight,channels, nWidth*nHeight*channels*sizeof(unsigned char));
		memset(pIMG->data_u8, 0x00, nWidth*nHeight*channels*sizeof(unsigned char));
		Vps_printf("ttt 27\n");
	}
	pIMG->dtype = 0;
	pIMG->size = nWidth*nHeight*channels*sizeof(unsigned char);
	pIMG->width = nWidth;
	pIMG->height = nHeight;
	pIMG->channels = channels;
	pIMG->step[0] = nWidth*channels;
}

void *AllocSpaceCR(int nSize)
{
	void *ptr = NULL;
	int align = 16;
	UTILS_assert(pOrigAddress != 0);
	ptr = (void*)(pOrigAddress+PointPos);
	nSize = ( ((size_t)nSize + align - 1) & ~(size_t)(align-1) );
	PointPos += nSize;
	UTILS_assert(PointPos < SpaceNum);
	return ptr;
}

void FreeSpaceCR(void *ptr)
{
	return;
}

void SetSpace(unsigned char* pAddress, int startAddress, int space)
{
	UTILS_assert(pAddress != 0);
	UTILS_assert(space > 0);
	pOrigAddress = pAddress;
	PointPos = startAddress;
	SpaceNum = space;
	Vps_printf("pAddress = 0x%08x, space = %d\n", pAddress, space);
	memset((unsigned char*)pAddress, 0, space);
}

void FreeIMGMat(IMG_MAT_FLOAT *pIMG)
{
	return;

	/*if(pIMG->data != NULL){
		FreeAlign(pIMG->data);
		pIMG->data = NULL;
	}
	memset(pIMG, 0, sizeof(IMG_MAT_FLOAT));*/
}

void FreeUcharMat(IMG_MAT_UCHAR *pIMG)
{
	return;

	/*if(pIMG->data != NULL){
		FreeAlign(pIMG->data);
		pIMG->data = NULL;
	}
	memset(pIMG, 0, sizeof(IMG_MAT_UCHAR));*/
}
#else
void AllocIMGMat(IMG_MAT_FLOAT *pIMG, int nWidth, int nHeight, int channels)
{
	if(/*pIMG->data == NULL*/1){
		pIMG->data  = (float*)MallocAlign(nWidth*nHeight*channels*sizeof(float));
		UTILS_assert(pIMG->data != NULL);
		memset(pIMG->data, 0x00, nWidth*nHeight*channels*sizeof(float));
	}
	pIMG->dtype = 1;
	pIMG->size = nWidth*nHeight*channels*sizeof(float);
	pIMG->width = nWidth;
	pIMG->height = nHeight;
	pIMG->channels = channels;
	pIMG->step[0] = nWidth*channels;
}

void AllocUcharMat(IMG_MAT_UCHAR *pIMG, int nWidth, int nHeight, int channels)
{
	if(/*pIMG->data == NULL*/1){
		pIMG->data_u8  = (unsigned char*)MallocAlign(nWidth*nHeight*channels*sizeof(unsigned char));
		UTILS_assert(pIMG->data_u8 != NULL);
		//Vps_printf("ttt 26 0x%08x %d %d %d %d\n",
		//		pIMG->data_u8, nWidth,nHeight,channels, nWidth*nHeight*channels*sizeof(unsigned char));
		//memset(pIMG->data_u8, 0x00, nWidth*nHeight*channels*sizeof(unsigned char));
		//Vps_printf("ttt 27\n");
	}
	pIMG->dtype = 0;
	pIMG->size = nWidth*nHeight*channels*sizeof(unsigned char);
	pIMG->width = nWidth;
	pIMG->height = nHeight;
	pIMG->channels = channels;
	pIMG->step[0] = nWidth*channels;
}

void FreeIMGMat(IMG_MAT_FLOAT *pIMG)
{
	if(pIMG->data != NULL){
		FreeAlign(pIMG->data);
		pIMG->data = NULL;
	}
	memset(pIMG, 0, sizeof(IMG_MAT_FLOAT));
}

void FreeUcharMat(IMG_MAT_UCHAR *pIMG)
{
	if(pIMG->data != NULL){
		FreeAlign(pIMG->data_u8);
		pIMG->data_u8 = NULL;
	}
	memset(pIMG, 0, sizeof(IMG_MAT_UCHAR));
}

void *AllocSpaceCR(int nSize)
{
	void *ptr = NULL;
	int align = 16;
	ptr = (void*)(unsigned char*)MallocAlign(nSize);
	UTILS_assert(ptr != NULL);
	return ptr;
}

void FreeSpaceCR(void *ptr)
{
	if(ptr == NULL){
		printf("WARN!!! ptr = NULL\n");
		return;
	}
	UTILS_assert(ptr != NULL);
	FreeAlign(ptr);
}

#endif

void matDump(int iStep, IMG_MAT_FLOAT *mat, int offset)
{
	char tag[][8] = {"A","B", "C", "D"};
	unsigned int *p;
	float *pf;
	p = (unsigned int *)mat->data;
	pf = (float*)p+offset;
	Vps_printf("%s == mat: %d,%d,%d ===\n", tag[iStep],
		mat->width, mat->height, mat->channels);
	Vps_printf("mat: %.6f %.6f %.6f %.6f\n",
		*pf, *(pf+1), *(pf+2), *(pf+3));
	Vps_printf("mat: %.6f %.6f %.6f %.6f\n",
		*(pf+4), *(pf+5), *(pf+6), *(pf+7));
	Vps_printf("mat: %.6f %.6f %.6f %.6f\n",
		*(pf+8), *(pf+9), *(pf+10), *(pf+11));
	Vps_printf("mat: %.6f %.6f %.6f %.6f\n",
		*(pf+12), *(pf+13), *(pf+14), *(pf+15));
}

void matDump_u8(int iStep, IMG_MAT_UCHAR *mat, int offset)
{
	char tag[][8] = {"A","B", "C", "D"};
	unsigned char *p, *pf;
	p = (unsigned char *)mat->data;
	pf = (unsigned char*)p+offset;
	Vps_printf("%s == mat: %d,%d,%d ===\n", tag[iStep],
		mat->width, mat->height, mat->channels);
	Vps_printf("mat: %08x %08x %08x %08x\n",
		*pf, *(pf+1), *(pf+2), *(pf+3));
	Vps_printf("mat: %08x %08x %08x %08x\n",
		*(pf+4), *(pf+5), *(pf+6), *(pf+7));
	Vps_printf("mat: %08x %08x %08x %08x\n",
		*(pf+8), *(pf+9), *(pf+10), *(pf+11));
	Vps_printf("mat: %08x %08x %08x %08x\n",
		*(pf+12), *(pf+13), *(pf+14), *(pf+15));
}

/**************************************/
void MergeUChar2Cplex(unsigned char *srcIMG, int srcWidth, int srcHeight, IMG_MAT_FLOAT *dstIMG)
{
	int i;
	ComplexCR	*pComplx;
	unsigned char *pSrcData;
	int nchannels = dstIMG->channels;

	UTILS_assert(srcWidth == dstIMG->width && srcHeight == dstIMG->height);

	pSrcData = srcIMG;
	pComplx = (ComplexCR	*)(dstIMG->data);
	#pragma UNROLL(4)
	for(i=0; i<srcWidth*srcHeight; i++)
	{
		pComplx[i].re = (float)pSrcData[i];
		pComplx[i].im = 0.f;

	}
}

void MergeFloat2Cplex(float *srcIMG, int srcWidth, int srcHeight, IMG_MAT_FLOAT *dstIMG)
{
	int i;
	ComplexCR	*pComplx;
	float *pSrcData;
	int nchannels = dstIMG->channels;

	UTILS_assert(srcWidth == dstIMG->width && srcHeight == dstIMG->height);
	UTILS_assert(dstIMG->channels == 2);

	pSrcData = srcIMG;
	pComplx = (ComplexCR	*)(dstIMG->data);
	memset(pComplx, 0, srcWidth*srcHeight*sizeof(ComplexCR));
	#pragma UNROLL(4)
	for(i=0; i<srcWidth*srcHeight; i++)
	{
		pComplx[i].re = pSrcData[i];
		//pComplx[i].im = 0.f;
	}
}

void SplitComplex(IMG_MAT_FLOAT *srcIMG, float *dst0, float *dst1, int dstWidth, int dstHeight)
{
	int k;
	ComplexCR	*pComplx;
	float *pdstData0, *pdstData1;
	int nchannels = srcIMG->channels;

	UTILS_assert(dstWidth == srcIMG->width && dstHeight == srcIMG->height);
	pComplx = (ComplexCR	*)(srcIMG->data);
	pdstData0 = dst0;
	pdstData1 = dst1;
	#pragma UNROLL(4)
	for(k=0; k<dstHeight*dstWidth; k++)
	{
		pdstData0[k] = pComplx[k].re;
		pdstData1[k] = pComplx[k].im;
	}
}

void realCR(IMG_MAT_FLOAT srcIMG, IMG_MAT_FLOAT *pReMat)
{
	int k;
	ComplexCR	*pComplx;
	int nchannels = srcIMG.channels;
	int dstHeight = pReMat->height;
	int dstWidth = pReMat->width;
	float *pdstData, *dst = pReMat->data;

	UTILS_assert(pReMat->width == srcIMG.width && pReMat->height == srcIMG.height);
	UTILS_assert(pReMat->channels == 1 && srcIMG.channels == 2);

	pdstData = dst;
	pComplx = (ComplexCR	*)(srcIMG.data);
	#pragma UNROLL(4)
	for(k=0; k<dstHeight*dstWidth; k++)
	{
		pdstData[k] = pComplx[k].re;
	}
}

void imagCR(IMG_MAT_FLOAT srcIMG, IMG_MAT_FLOAT *pImMat)
{
	int k;
	ComplexCR	*pComplx;
	int nchannels = srcIMG.channels;
	int dstHeight = pImMat->height;
	int dstWidth = pImMat->width;
	float *pdstData, *dst = pImMat->data;

	UTILS_assert(pImMat->width == srcIMG.width && pImMat->height == srcIMG.height);
	UTILS_assert(pImMat->channels == 1 && srcIMG.channels == 2);
	pdstData = dst;
	pComplx = (ComplexCR	*)(srcIMG.data);
	#pragma UNROLL(4)
	for(k=0; k<dstHeight*dstWidth; k++)
	{
		pdstData[k] = pComplx[k].im;
	}

}

void rearrangeCR(IMG_MAT_FLOAT *src, IMG_MAT_FLOAT midMat)
{
	int y;
	int nWidth = src->width;
	int nHeight = src->height;
	float *A, *B, *C, *D;
	float *T;

	UTILS_assert(src->width == midMat.width && src->height == midMat.height);
	UTILS_assert(src->channels == midMat.channels);

	A = (src->data);
	B = (src->data + src->step[0]/2);
	C = (src->data + src->step[0] * nHeight/2);
	D = (src->data + src->step[0] * nHeight/2 + src->step[0]/2);
	T = midMat.data;
	#pragma UNROLL(4)
	for(y=0; y<nHeight/2; y++)
	{
		memcpy(T, D, src->step[0]*sizeof(float)/2);
		memcpy(D, A, src->step[0]*sizeof(float)/2);
		memcpy(A, T, src->step[0]*sizeof(float)/2);
	
		memcpy(T, C, src->step[0]*sizeof(float)/2);
		memcpy(C, B, src->step[0]*sizeof(float)/2);
		memcpy(B, T, src->step[0]*sizeof(float)/2);

		A += src->step[0];
		B += src->step[0];
		C += src->step[0];
		D += src->step[0];
	}
}

void complexMultiCR(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, IMG_MAT_FLOAT dst)
{
	mulSpectrumsCR(src0, src1, dst, 0, 0);
}

void complexDivisionCR(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, IMG_MAT_FLOAT dst)
{
	int nWidth = src0.width;
	int nHeight = src0.height;
	int nChannel = src0.channels;
	ComplexCR *pSrc0, *pSrc1, *pDst;
	int k;

	UTILS_assert(src0.width == dst.width && src0.height == dst.height);
	UTILS_assert(src1.width == dst.width && src1.height == dst.height);
	UTILS_assert(src0.channels == 2 && src1.channels == 2 && dst.channels == 2);

	pSrc0 = (ComplexCR *)(src0.data);
	pSrc1 = (ComplexCR *)(src1.data);
	pDst = (ComplexCR *)(dst.data);
	#pragma UNROLL(4)	
	for(k=0; k<nWidth*nHeight; k++)
	{
		double mag = ((double)pSrc1[k].re*pSrc1[k].re + (double)pSrc1[k].im*pSrc1[k].im);
//		assert(mag > 1e-15);
		double t = 1./mag;
		double re = (double)((pSrc0[k].re*pSrc1[k].re + pSrc0[k].im*pSrc1[k].im)*t);
		double im = (double)((-pSrc0[k].re*pSrc1[k].im + pSrc0[k].im*pSrc1[k].re)*t);
		pDst[k].re  = (float)re;
		pDst[k].im  = (float)im;
	}
}

void AddMat(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, IMG_MAT_FLOAT dst)
{
	int nWidth = src0.width;
	int nHeight = src0.height;
	int nChannel = src0.channels;
	float *pSrc0, *pSrc1, *pDst;
	int k;

	UTILS_assert(src0.width == dst.width && src0.height == dst.height);
	UTILS_assert(src1.width == dst.width && src1.height == dst.height);
	UTILS_assert(src0.channels == src1.channels && src1.channels == dst.channels);
	pSrc0 = (float *)(src0.data);
	pSrc1 = (float *)(src1.data);
	pDst = (float *)(dst.data);
	#pragma UNROLL(4)
	for(k=0; k<nWidth*nHeight; k++)
	{
		pDst[k] = pSrc0[k] + pSrc1[k];
	}
}

void AddK(IMG_MAT_FLOAT src, float K, IMG_MAT_FLOAT dst)
{
	int nWidth = src.width;
	int nHeight = src.height;
	int nChannel = src.channels;
	float *pSrc,  *pDst;
	ComplexCR *pSrcCex, *pDstCex;
	int k;

	UTILS_assert(src.width == dst.width && src.height == dst.height);
	UTILS_assert(src.channels == dst.channels);
	if(nChannel == 1){
		pSrc = (float *)(src.data);
		pDst = (float *)(dst.data);
		#pragma UNROLL(4)
		for(k=0; k<nWidth*nHeight; k++)
		{
			pDst[k] = pSrc[k] + K;
		}
	}else if(nChannel == 2){
		pSrcCex = (ComplexCR *)(src.data);
		pDstCex = (ComplexCR *)(dst.data);
		#pragma UNROLL(4)
		for(k=0; k<nWidth*nHeight; k++)
		{
		
			pDstCex[k].re = pSrcCex[k].re + K;
			pDstCex[k].im = pSrcCex[k].im;
		}
	}
}

void ExpMat(IMG_MAT_FLOAT src, IMG_MAT_FLOAT dst, float sigma)
{
	int nWidth = src.width;
	int nHeight = src.height;
	int nChannel = src.channels;
	float *pSrc,  *pDst;
	int  k;
	float delta = 1.f/sigma*sigma;

	UTILS_assert(src.width == dst.width && src.height == dst.height);
	UTILS_assert(src.channels == 1 && dst.channels == 1);

	pSrc = (float *)(src.data);
	pDst = (float *)(dst.data);
	#pragma UNROLL(4)
	for(k=0; k<nWidth*nHeight; k++)
	{
		pDst[k] = exp(-pSrc[k]*delta);
	}
}

void MinMaxLoc(IMG_MAT_FLOAT src, float* _minval, float* _maxval, PointICR* _minidx, PointICR* _maxidx )
{
	int i, j, k;
	int nWidth = src.width;
	int nHeight = src.height;
	float fminval = 10000.0f;//FLT_MAX, 
	float fmaxval = -10000.0f;//FLT_MAX;
	float *pSrc;
	PointICR		minIdx, maxIdx;

	UTILS_assert(src.channels == 1);
	pSrc = (float *)(src.data);
	k = 0;
	for(j=0; j<nHeight; j++)
	{
		for(i=0; i<nWidth; i++)
		{
			if(pSrc[k] <fminval){
				fminval = pSrc[k];
				minIdx.x = i;		minIdx.y = j;
			}
			if(pSrc[k] > fmaxval){
				fmaxval = pSrc[k];
				maxIdx.x = i;		maxIdx.y = j;
			}
			k++;
		}
	}
	if(_minval != NULL)
		*_minval = fminval;
	if(_maxval != NULL)
		*_maxval = fmaxval;
	if(_minidx != NULL)
		*_minidx = minIdx;
	if(_maxidx != NULL)
		*_maxidx = maxIdx;
}

float SubPixelPeak(float left, float center, float right)
{   
	float divisor = 2 * center - right - left;

	if (divisor == 0)
		return 0;

	return 0.5 * (right - left) / divisor;
}

int CalTgtStruct(IMG_MAT_UCHAR image, UTC_Rect roiRc, int *SSIM)
{
	int i, j;
	int xbegin, xend, ybegin, yend;
	int xwidth, yheight;
	int width, height;
	unsigned char *lpFrameY;
	int meanv = 0, vsquared = 0;

	xbegin = roiRc.x;
	xend = roiRc.x + roiRc.width;
	ybegin = roiRc.y;
	yend = roiRc.y + roiRc.height;
	xwidth = roiRc.width;
	yheight = roiRc.height;
	width = image.width;
	height = image.height;

	lpFrameY = (unsigned char*)image.data_u8;
	for (j=ybegin; j<yend; j++){
#pragma UNROLL(4)
		for(i=xbegin; i<xend; i++){
			meanv += lpFrameY[j*width+i];
		}
	}
	meanv /= (xwidth*yheight);
	for (j=ybegin; j<yend; j++){
#pragma UNROLL(4)
		for(i=xbegin; i<xend; i++){
			vsquared += (lpFrameY[j*width+i]-meanv)*(lpFrameY[j*width+i]-meanv);
		}
	}
	vsquared /= (xwidth*yheight);
	vsquared = (int)(sqrt((double)vsquared));
	if(SSIM != NULL){
		*SSIM  = vsquared;
	}
	return vsquared;
}

static const int SMALL_GAUSSIAN_SIZE = 7;
static const float small_gaussian_tab[4][7] =
{
	{1.f},
	{0.25f, 0.5f, 0.25f},
	{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
	{0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
};
static float gaussian2DTab[7][7] ={0,};
static UTC_SIZE gaussScale = {0, 0};

static void GaussCoefInit(UTC_SIZE kelSize)
{
	int i, j;
	int Idx = kelSize.width/2;
	float sum = 0.f;
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
}

void GaussianFilterCR(IMG_MAT_UCHAR *psrc, IMG_MAT_UCHAR *pdst,  UTC_Rect roiRect, UTC_SIZE kelSize)
{
	int i, j, x, y;
	int width = psrc->width;
	int height = psrc->height;
	int kelArean = kelSize.width*kelSize.height;
	Uint8 *psrcdata, *pdstdata,*pmiddata;
	float sum = 0.f;

	assert(kelSize.width == kelSize.height && (kelSize.width == 3 || kelSize.width  == 5 || kelSize.width == 7));
	assert(psrc != NULL && pdst != NULL);
	assert(psrc->width == pdst->width && psrc->height == pdst->height);

	if(gaussScale.width != kelSize.width || gaussScale.height != kelSize.height){
		GaussCoefInit(kelSize);
		gaussScale = kelSize;
	}

	for(y=roiRect.y+kelSize.height/2; y<roiRect.y+roiRect.height-kelSize.height/2; y++)
	{
		for(x=roiRect.x+kelSize.width/2; x<roiRect.x+roiRect.width - kelSize.width/2; x++)
		{
			psrcdata = psrc->data_u8 + y*psrc->width + x;
			pdstdata = pdst->data_u8 + y*pdst->width + x;
			sum = 0.f;
			for(j=0; j<kelSize.height; j++)
			{
				pmiddata = psrcdata + (j-kelSize.height/2)*psrc->width - kelSize.width/2;
				for(i=0; i<kelSize.width; i++)
				{
					sum += pmiddata[i]*gaussian2DTab[j][i];
				}
			}
			pdstdata[0] = (Uint8)sum;
		}
	}
}

void BlurCR(IMG_MAT_UCHAR *psrc, IMG_MAT_UCHAR *pdst,  UTC_Rect roiRect, UTC_SIZE kelSize)
{
	int i, j, x, y;
	int width = psrc->width;
	int height = psrc->height;
	int kelArean = kelSize.width*kelSize.height;
	Uint8 *psrcdata, *pdstdata, *pmiddata;
	float sum;
	assert(psrc != NULL && pdst != NULL);
	assert(psrc->width == pdst->width && psrc->height == pdst->height);

	for(y=roiRect.y+kelSize.height/2; y<roiRect.y+roiRect.height-kelSize.height/2; y++)
	{
		for(x=roiRect.x+kelSize.width/2; x<roiRect.x+roiRect.width - kelSize.width/2; x++)
		{
			psrcdata = psrc->data_u8 + y*psrc->width + x;
			pdstdata = pdst->data_u8 + y*pdst->width + x;
			sum = 0.f;
			for(j=0; j<kelSize.height; j++)
			{
				pmiddata = psrcdata + (j-kelSize.height/2)*psrc->width - kelSize.width/2;
				for(i=0; i<kelSize.width; i++)
				{
					sum += pmiddata[i];
				}
			}
			pdstdata[0] =  (Uint8)(sum/kelArean);
		}
	}
}


