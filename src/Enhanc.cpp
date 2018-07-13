#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "Enhance.h"
#include <omp.h>
#include "arm_neon.h"
#include "SSE2NEON.h"

/*********************** Local prototypes ************************/
static void ClipHistogram (unsigned int*, unsigned int, unsigned int);
static void MakeHistogram (kz_pixel_t*, unsigned int, unsigned int, unsigned int,
			unsigned int*, unsigned int, kz_pixel_t*);
static void MapHistogram (unsigned int*, kz_pixel_t, kz_pixel_t,
	       unsigned int, unsigned int);
static void MakeLut (kz_pixel_t*, kz_pixel_t, kz_pixel_t, unsigned int);
static void Interpolate (kz_pixel_t*, int, unsigned int*, unsigned int*,
			unsigned int*, unsigned int*, unsigned int, unsigned int, kz_pixel_t*);
static	void Interpolate_Neon(kz_pixel_t * pImage, int uiXRes, unsigned char * pulMapLU,
        unsigned char * , unsigned char * ,  unsigned char * , unsigned int , unsigned int , kz_pixel_t* );

const unsigned int uiMAX_REG_X = 16;	  /* max. # contextual regions in x-direction */
const unsigned int uiMAX_REG_Y = 16;	  /* max. # contextual regions in y-direction */

/************************** main function CLAHE ******************/
int CLAHE_enh (kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
		   unsigned int startX, unsigned int startY, unsigned int uiWidth, unsigned int uiHeight,
           kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
           unsigned int uiNrBins, float fCliplimit)
           /*  pImage - Pointer to the input/output image
           *   uiXRes - Image resolution in the X direction
           *   uiYRes - Image resolution in the Y direction
           *   Min - Minimum greyvalue of input image (also becomes minimum of output image)
           *   Max - Maximum greyvalue of input image (also becomes maximum of output image)
           *   uiNrX - Number of contextial regions in the X direction (min 2, max uiMAX_REG_X)
           *   uiNrY - Number of contextial regions in the Y direction (min 2, max uiMAX_REG_Y)
           *   uiNrBins - Number of greybins for histogram ("dynamic range")
           *   float fCliplimit - Normalized cliplimit (higher values give more contrast)
           *
           */
{
    unsigned int uiX, uiY;        /* counters */
    unsigned int uiXSize, uiYSize, uiSubX, uiSubY; /* size of context. reg. and subimages */
    unsigned int uiXL, uiXR, uiYU, uiYB;  /* auxiliary variables interpolation routine */
    unsigned int ulClipLimit, ulNrPixels;/* clip limit and region pixel count */
    kz_pixel_t* pImPointer;        /* pointer to image */
    kz_pixel_t aLUT[uiNR_OF_GREY];      /* lookup table used for scaling of input image */
    unsigned int* pulHist, *pulMapArray; /* pointer to histogram and mappings*/
    unsigned int* pulLU, *pulLB, *pulRU, *pulRB; /* auxiliary pointers interpolation */

    if (uiNrX > uiMAX_REG_X) return -1;     /* # of regions x-direction too large */
    if (uiNrY > uiMAX_REG_Y) return -2;     /* # of regions y-direction too large */

    if (uiWidth % uiNrX) return -3;      /* x-resolution no multiple of uiNrX */
    if (uiHeight % uiNrY) return -4;      /* y-resolution no multiple of uiNrY */

    if (Max >= uiNR_OF_GREY) return -5;     /* maximum too large */
    if (Min >= Max) return -6;         /* minimum equal or larger than maximum */
    if (uiNrX < 2 || uiNrY < 2) return -7;/* at least 4 contextual regions required */
    if (fCliplimit == 1.0) return 0;      /* is OK, immediately returns original image. */
    if (uiNrBins == 0) uiNrBins = 128;    /* default value when not specified */

    pulMapArray=(unsigned int *)malloc(sizeof(unsigned int)*uiNrX*uiNrY*uiNrBins);
    if (pulMapArray == 0) return -8;      /* Not enough memory! (try reducing uiNrBins) */

    uiXSize = uiWidth/uiNrX; uiYSize = uiHeight/uiNrY;  /* Actual size of contextual regions */
    ulNrPixels = (unsigned int)uiXSize * (unsigned int)uiYSize;

    if(fCliplimit > 0.0) {         /* Calculate actual cliplimit  */
        ulClipLimit = (unsigned int) (fCliplimit * (uiXSize * uiYSize) / uiNrBins);
        ulClipLimit = (ulClipLimit < 1UL) ? 1UL : ulClipLimit;
    }
    else ulClipLimit = 1UL<<14;         /* Large value, do not clip (AHE) */
    MakeLut(aLUT, Min, Max, uiNrBins);    /* Make lookup table for mapping of greyvalues */
    /* Calculate greylevel mappings for each contextual region */

    for (uiY = 0, pImPointer = (pImage+startY*uiXRes+startX); uiY < uiNrY; uiY++)
    {
        for (uiX = 0; uiX < uiNrX; uiX++, pImPointer += uiXSize)
        {
            pulHist = &pulMapArray[uiNrBins * (uiY * uiNrX + uiX)];
            MakeHistogram(pImPointer,uiXRes,uiXSize,uiYSize,pulHist,uiNrBins,aLUT);
            ClipHistogram(pulHist, uiNrBins, ulClipLimit);
            MapHistogram(pulHist, Min, Max, uiNrBins, ulNrPixels);
        }
        pImPointer	+= (uiXRes - uiWidth);
        pImPointer += (uiYSize - 1) * uiXRes;         /* skip lines, set pointer */
    }

    /* Interpolate greylevel mappings to get CLAHE image */
    for (pImPointer = (pImage+startY*uiXRes+startX), uiY = 0; uiY <= uiNrY; uiY++)
    {
        if (uiY == 0)
        {                     /* special case: top row */
            uiSubY = uiYSize >> 1;  uiYU = 0; uiYB = 0;
        }
        else
        {
            if (uiY == uiNrY) {               /* special case: bottom row */
                uiSubY = uiYSize >> 1;    uiYU = uiNrY-1;  uiYB = uiYU;
            }
            else
            {                     /* default values */
                uiSubY = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
            }
        }
        for (uiX = 0; uiX <= uiNrX; uiX++)
        {
            if (uiX == 0)
            {                 /* special case: left column */
                uiSubX = uiXSize >> 1; uiXL = 0; uiXR = 0;
            }
            else
            {
                if (uiX == uiNrX)
                {             /* special case: right column */
                    uiSubX = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
                }
                else
                {                     /* default values */
                    uiSubX = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
                }
            }

            pulLU = &pulMapArray[uiNrBins * (uiYU * uiNrX + uiXL)];
            pulRU = &pulMapArray[uiNrBins * (uiYU * uiNrX + uiXR)];
            pulLB = &pulMapArray[uiNrBins * (uiYB * uiNrX + uiXL)];
            pulRB = &pulMapArray[uiNrBins * (uiYB * uiNrX + uiXR)];
            Interpolate(pImPointer,uiXRes,pulLU,pulRU,pulLB,pulRB,uiSubX,uiSubY,aLUT);
            pImPointer += uiSubX;             /* set pointer on next matrix */
        }
        pImPointer	+= (uiXRes - uiWidth);
        pImPointer += (uiSubY - 1) * uiXRes;
    }

    free(pulMapArray);                    /* free space for histograms */

    return 0;                         /* return status OK */
}

int CLAHE_enh_omp (kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
			unsigned int startX, unsigned int startY, unsigned int uiWidth, unsigned int uiHeight,
           kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
           unsigned int uiNrBins, float fCliplimit)
{
    unsigned int uiX, uiY;        /* counters */
    unsigned int uiXSize, uiYSize, uiSubX, uiSubY; /* size of context. reg. and subimages */
    unsigned int uiXL, uiXR, uiYU, uiYB;  /* auxiliary variables interpolation routine */
    unsigned int ulClipLimit, ulNrPixels;/* clip limit and region pixel count */
    kz_pixel_t* pImPointer, * pImPtr[MAX_BLOCKS_NUM];        /* pointer to image */
    kz_pixel_t aLUT[uiNR_OF_GREY];      /* lookup table used for scaling of input image */
    unsigned int*pulMapArray; /* pointer to histogram and mappings*/
    unsigned int* pulLU[MAX_BLOCKS_NUM], *pulLB[MAX_BLOCKS_NUM], *pulRU[MAX_BLOCKS_NUM], *pulRB[MAX_BLOCKS_NUM]; /* auxiliary pointers interpolation */
    unsigned int* pulHist[MAX_BLOCKS_NUM];
	unsigned int uiSubW[MAX_BLOCKS_NUM], uiSubH[MAX_BLOCKS_NUM];
    int k,	calIdx = 0;

    if (uiNrX > uiMAX_REG_X) return -1;     /* # of regions x-direction too large */
    if (uiNrY > uiMAX_REG_Y) return -2;     /* # of regions y-direction too large */

    if (uiWidth % uiNrX) return -3;      /* x-resolution no multiple of uiNrX */
    if (uiHeight % uiNrY) return -4;      /* y-resolution no multiple of uiNrY */

    if (Max >= uiNR_OF_GREY) return -5;     /* maximum too large */
    if (Min >= Max) return -6;         /* minimum equal or larger than maximum */
    if (uiNrX < 2 || uiNrY < 2) return -7;/* at least 4 contextual regions required */
    if (fCliplimit == 1.0) return 0;      /* is OK, immediately returns original image. */
    if (uiNrBins == 0) uiNrBins = 128;    /* default value when not specified */

    if(uiNrX*uiNrY > MAX_BLOCKS_NUM)	return -9;	/* blocks number are too many*/

    pulMapArray=(unsigned int *)malloc(sizeof(unsigned int)*uiNrX*uiNrY*uiNrBins);
    if (pulMapArray == 0) return -8;      /* Not enough memory! (try reducing uiNrBins) */

    uiXSize = uiWidth/uiNrX; uiYSize = uiHeight/uiNrY;  /* Actual size of contextual regions */
    ulNrPixels = (unsigned int)uiXSize * (unsigned int)uiYSize;

    if(fCliplimit > 0.0) {         /* Calculate actual cliplimit  */
        ulClipLimit = (unsigned int) (fCliplimit * (uiXSize * uiYSize) / uiNrBins);
        ulClipLimit = (ulClipLimit < 1UL) ? 1UL : ulClipLimit;
    }
    else ulClipLimit = 1UL<<14;         /* Large value, do not clip (AHE) */
    MakeLut(aLUT, Min, Max, uiNrBins);    /* Make lookup table for mapping of greyvalues */
    /* Calculate greylevel mappings for each contextual region */

    calIdx = 0;
    for (uiY = 0, pImPointer = (pImage+startY*uiXRes+startX); uiY < uiNrY; uiY++)
   {
	   for (uiX = 0; uiX < uiNrX; uiX++, pImPointer += uiXSize)
	   {
		   pulHist[calIdx] = &pulMapArray[uiNrBins * (uiY * uiNrX + uiX)];
		   pImPtr[calIdx] = pImPointer;
		   calIdx++;
	   }
	   pImPointer	+= (uiXRes - uiWidth);
	   pImPointer += (uiYSize - 1) * uiXRes;         /* skip lines, set pointer */
   }
#pragma omp parallel for
    for(k=0; k<calIdx; k++)
    {
    	MakeHistogram(pImPtr[k],uiXRes,uiXSize,uiYSize,pulHist[k],uiNrBins,aLUT);
    	ClipHistogram(pulHist[k], uiNrBins, ulClipLimit);
    	MapHistogram(pulHist[k], Min, Max, uiNrBins, ulNrPixels);
    }

    calIdx = 0;
    /* Interpolate greylevel mappings to get CLAHE image */
	for (pImPointer = (pImage+startY*uiXRes+startX), uiY = 0; uiY <= uiNrY; uiY++)
	{
		if (uiY == 0)
		{                     /* special case: top row */
			uiSubY = uiYSize >> 1;  uiYU = 0; uiYB = 0;
		}
		else
		{
			if (uiY == uiNrY) {               /* special case: bottom row */
				uiSubY = uiYSize >> 1;    uiYU = uiNrY-1;  uiYB = uiYU;
			}
			else
			{                     /* default values */
				uiSubY = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
			}
		}
		for (uiX = 0; uiX <= uiNrX; uiX++)
		{
			if (uiX == 0)
			{                 /* special case: left column */
				uiSubX = uiXSize >> 1; uiXL = 0; uiXR = 0;
			}
			else
			{
				if (uiX == uiNrX)
				{             /* special case: right column */
					uiSubX = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
				}
				else
				{                     /* default values */
					uiSubX = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
				}
			}

			pulLU[calIdx] = &pulMapArray[uiNrBins * (uiYU * uiNrX + uiXL)];
			pulRU[calIdx] = &pulMapArray[uiNrBins * (uiYU * uiNrX + uiXR)];
			pulLB[calIdx] = &pulMapArray[uiNrBins * (uiYB * uiNrX + uiXL)];
			pulRB[calIdx] = &pulMapArray[uiNrBins * (uiYB * uiNrX + uiXR)];
			uiSubW[calIdx] = uiSubX;
			uiSubH[calIdx] = uiSubY;
			pImPtr[calIdx] = pImPointer;

			pImPointer += uiSubX;             /* set pointer on next matrix */
			calIdx++;
		}
		pImPointer	+= (uiXRes - uiWidth);
		pImPointer += (uiSubY - 1) * uiXRes;
	}
#pragma omp parallel for
	for(k=0; k<calIdx; k++)
	{
		Interpolate(pImPtr[k],uiXRes,pulLU[k],pulRU[k],pulLB[k],pulRB[k],uiSubW[k],uiSubH[k],aLUT);
	}

    free(pulMapArray);                    /* free space for histograms */
    return 0;                         /* return status OK */
}

void ClipHistogram (unsigned int* pulHistogram, unsigned int
                    uiNrGreylevels, unsigned int ulClipLimit)
{
    unsigned int* pulBinPointer, *pulEndPointer, *pulHisto;
    unsigned int ulNrExcess, ulOldNrExcess, ulUpper, ulBinIncr, ulStepSize, i;
    long lBinExcess;

    ulNrExcess = 0;  pulBinPointer = pulHistogram;
    for (i = 0; i < uiNrGreylevels; i++)
    { /* calculate total number of excess pixels */
        lBinExcess = (int) pulBinPointer[i] - (int) ulClipLimit;
        if (lBinExcess > 0) ulNrExcess += lBinExcess;      /* excess in current bin */
    };

    /* Second part: clip histogram and redistribute excess pixels in each bin */
    ulBinIncr = ulNrExcess / uiNrGreylevels;          /* average binincrement */
    ulUpper =  ulClipLimit - ulBinIncr;  /* Bins larger than ulUpper set to cliplimit */

    for (i = 0; i < uiNrGreylevels; i++)
    {
        if (pulHistogram[i] > ulClipLimit)
            pulHistogram[i] = ulClipLimit; /* clip bin */
        else
        {
            if (pulHistogram[i] > ulUpper)
            {       /* high bin count */
 //               ulNrExcess -= (pulHistogram[i] - ulUpper); pulHistogram[i]=ulClipLimit;
				ulNrExcess -= (ulClipLimit -pulHistogram[i]); pulHistogram[i]=ulClipLimit;
            }
            else
            {                   /* low bin count */
                ulNrExcess -= ulBinIncr; pulHistogram[i] += ulBinIncr;
            }
        }
    }

    do {   /* Redistribute remaining excess  */
        pulEndPointer = &pulHistogram[uiNrGreylevels]; pulHisto = pulHistogram;

        ulOldNrExcess = ulNrExcess;     /* Store number of excess pixels for test later. */

        while (ulNrExcess && pulHisto < pulEndPointer)
        {
            ulStepSize = uiNrGreylevels / ulNrExcess;
            if (ulStepSize < 1)
                ulStepSize = 1;       /* stepsize at least 1 */
            for (pulBinPointer=pulHisto; pulBinPointer < pulEndPointer && ulNrExcess; pulBinPointer += ulStepSize)
            {
                if (*pulBinPointer < ulClipLimit)
                {
                    (*pulBinPointer)++;  ulNrExcess--;    /* reduce excess */
                }
            }
            pulHisto++;       /* restart redistributing on other bin location */
        }
    } while ((ulNrExcess) && (ulNrExcess < ulOldNrExcess));
    /* Finish loop when we have no more pixels or we can't redistribute any more pixels */
}

void MakeHistogram (kz_pixel_t* pImage, unsigned int uiXRes,
                    unsigned int uiSizeX, unsigned int uiSizeY,
                    unsigned int* pulHistogram,
                    unsigned int uiNrGreylevels, kz_pixel_t* pLookupTable)
{
    kz_pixel_t* pImagePointer;
    unsigned int i;

    for (i = 0; i < uiNrGreylevels; i++)
        pulHistogram[i] = 0L; /* clear histogram */

    for (i = 0; i < uiSizeY; i++)
    {
        pImagePointer = &pImage[uiSizeX];
        while (pImage < pImagePointer)
            pulHistogram[pLookupTable[*pImage++]]++;
        pImagePointer += uiXRes;
        pImage = pImagePointer-uiSizeX;
    }
}

void MapHistogram (unsigned int* pulHistogram, kz_pixel_t Min, kz_pixel_t Max,
                   unsigned int uiNrGreylevels, unsigned int ulNrOfPixels)
{
    unsigned int i;  unsigned int ulSum = 0;
    const float fScale = ((float)(Max - Min)) / ulNrOfPixels;
    const unsigned int ulMin = (unsigned int) Min;

    for (i = 0; i < uiNrGreylevels; i++)
    {
        ulSum += pulHistogram[i];
        pulHistogram[i]=(unsigned int)(ulMin+ulSum*fScale);
        pulHistogram[i] = (pulHistogram[i] > Max)?Max:pulHistogram[i];
    }
}

void MapHistogramU8 (unsigned int* pulHistogram,unsigned char* pulU8Histogram, kz_pixel_t Min, kz_pixel_t Max,
                   unsigned int uiNrGreylevels, unsigned int ulNrOfPixels)
{
    unsigned int i;  unsigned int ulSum = 0;
    const float fScale = ((float)(Max - Min)) / ulNrOfPixels;
    const unsigned int ulMin = (unsigned int) Min;

    for (i = 0; i < uiNrGreylevels; i++)
    {
        ulSum += pulHistogram[i];
        pulHistogram[i]=(unsigned int)(ulMin+ulSum*fScale);
        pulU8Histogram[i] =(unsigned char)( (pulHistogram[i] > Max)?Max:pulHistogram[i]);
    }
}

void MakeLut (kz_pixel_t * pLUT, kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrBins)
{
    int i;
    const kz_pixel_t BinSize = (kz_pixel_t) (1 + (Max - Min) / uiNrBins);

    for (i = Min; i <= Max; i++)
        pLUT[i] = (i - Min) / BinSize;
}

void Interpolate (kz_pixel_t * pImage, int uiXRes, unsigned int * pulMapLU,
                  unsigned int * pulMapRU, unsigned int * pulMapLB,  unsigned int * pulMapRB,
                  unsigned int uiXSize, unsigned int uiYSize, kz_pixel_t* pLUT)
                  /* pImage      - pointer to input/output image
                  * uiXRes      - resolution of image in x-direction
                  * pulMap*     - mappings of greylevels from histograms
                  * uiXSize     - uiXSize of image submatrix
                  * uiYSize     - uiYSize of image submatrix
                  * pLUT           - lookup table containing mapping greyvalues to bins
                  */
{
    const unsigned int uiIncr = uiXRes-uiXSize; /* Pointer increment after processing row */
    kz_pixel_t GreyValue; unsigned int uiNum = uiXSize*uiYSize; /* Normalization factor */

    unsigned int uiXCoef, uiYCoef, uiXInvCoef, uiYInvCoef, uiShift = 0;

    if (uiNum & (uiNum - 1))   /* If uiNum is not a power of two, use division */
	{
        for (uiYCoef = 0, uiYInvCoef = uiYSize; uiYCoef < uiYSize;  uiYCoef++, uiYInvCoef--,pImage+=uiIncr)
        {
            for (uiXCoef = 0, uiXInvCoef = uiXSize; uiXCoef < uiXSize;  uiXCoef++, uiXInvCoef--)
            {
                GreyValue = (unsigned char)pLUT[(unsigned char)(*pImage)];         /* get histogram bin value */
                *pImage++ = (kz_pixel_t ) ((uiYInvCoef * (uiXInvCoef*pulMapLU[GreyValue] + uiXCoef * pulMapRU[GreyValue])
										  + uiYCoef * (uiXInvCoef * pulMapLB[GreyValue] + uiXCoef * pulMapRB[GreyValue])) / uiNum);
            }
        }
	}
	else  /* avoid the division and use a right shift instead */
	{
		while (uiNum >>= 1) uiShift++;           /* Calculate 2log of uiNum */
		for (uiYCoef = 0, uiYInvCoef = uiYSize; uiYCoef < uiYSize;  uiYCoef++, uiYInvCoef--,pImage+=uiIncr)
		{
			for (uiXCoef = 0, uiXInvCoef = uiXSize; uiXCoef < uiXSize;  uiXCoef++, uiXInvCoef--)
			{
				GreyValue = pLUT[*pImage];    /* get histogram bin value */
				*pImage++ = (kz_pixel_t)((uiYInvCoef* (uiXInvCoef * pulMapLU[GreyValue]	+ uiXCoef * pulMapRU[GreyValue])
										+ uiYCoef * (uiXInvCoef * pulMapLB[GreyValue] + uiXCoef * pulMapRB[GreyValue])) >> uiShift);
			}
		}
	}
}

static void InterpolateU8 (kz_pixel_t * pImage, int uiXRes, unsigned char * pulMapLU,
                  unsigned char * pulMapRU, unsigned char * pulMapLB,  unsigned char * pulMapRB,
                  unsigned int uiXSize, unsigned int uiYSize, kz_pixel_t* pLUT)
                  /* pImage      - pointer to input/output image
                  * uiXRes      - resolution of image in x-direction
                  * pulMap*     - mappings of greylevels from histograms
                  * uiXSize     - uiXSize of image submatrix
                  * uiYSize     - uiYSize of image submatrix
                  * pLUT           - lookup table containing mapping greyvalues to bins
                  */
{
    const unsigned int uiIncr = uiXRes-uiXSize; /* Pointer increment after processing row */
    kz_pixel_t GreyValue; unsigned int uiNum = uiXSize*uiYSize; /* Normalization factor */

    unsigned int uiXCoef, uiYCoef, uiXInvCoef, uiYInvCoef, uiShift = 0;

    if (uiNum & (uiNum - 1))   /* If uiNum is not a power of two, use division */
	{
        for (uiYCoef = 0, uiYInvCoef = uiYSize; uiYCoef < uiYSize;  uiYCoef++, uiYInvCoef--,pImage+=uiIncr)
        {
            for (uiXCoef = 0, uiXInvCoef = uiXSize; uiXCoef < uiXSize;  uiXCoef++, uiXInvCoef--)
            {
                GreyValue = (unsigned char)pLUT[(unsigned char)(*pImage)];         /* get histogram bin value */
                *pImage++ = (kz_pixel_t ) ((uiYInvCoef * (uiXInvCoef*pulMapLU[GreyValue] + uiXCoef * pulMapRU[GreyValue])
										  + uiYCoef * (uiXInvCoef * pulMapLB[GreyValue] + uiXCoef * pulMapRB[GreyValue])) / uiNum);
            }
        }
	}
	else  /* avoid the division and use a right shift instead */
	{
		while (uiNum >>= 1) uiShift++;           /* Calculate 2log of uiNum */
		for (uiYCoef = 0, uiYInvCoef = uiYSize; uiYCoef < uiYSize;  uiYCoef++, uiYInvCoef--,pImage+=uiIncr)
		{
			for (uiXCoef = 0, uiXInvCoef = uiXSize; uiXCoef < uiXSize;  uiXCoef++, uiXInvCoef--)
			{
				GreyValue = pLUT[*pImage];    /* get histogram bin value */
				*pImage++ = (kz_pixel_t)((uiYInvCoef* (uiXInvCoef * pulMapLU[GreyValue]	+ uiXCoef * pulMapRU[GreyValue])
										+ uiYCoef * (uiXInvCoef * pulMapLB[GreyValue] + uiXCoef * pulMapRB[GreyValue])) >> uiShift);
			}
		}
	}
}

static void interp_neon(unsigned char * pImage, unsigned int *pCoefXTbl, unsigned int *pCoefInvXTbl, unsigned int  uiYCoef, unsigned int uiXSize, unsigned int uiYSize, unsigned int uiNum,
		unsigned char * pulMapLU,unsigned char * pulMapRU, unsigned char * pulMapLB,  unsigned char * pulMapRB,unsigned char* pLUT)
{
	unsigned int uiXCoef, uiXInvCoef, uiYInvCoef;
	unsigned char *pImgPtr;
	uint8x8_t v11, v12,   v21, v22;
	uint8_t __attribute__((aligned(16)))		grayv[8], grayv11[8], grayv12[8], grayv21[8], grayv22[8];
	int dim8 = (uiXSize >> 3);
	int left8 = (uiXSize & 7);
	int k;
	kz_pixel_t GreyValue;

	unsigned int intY = uiYCoef*4096/uiYSize;
	unsigned int one_y = 4096-intY;
	uint32_t intY_32 = (uint32_t) intY;
	uint32_t oneY_32 = (uint32_t) one_y;

	uint32x4_t	coefX_l,	coefX_h, coefX_inv_l, coefX_inv_h;
	uiYInvCoef = uiYSize - uiYCoef;
	uiXCoef = 0, uiXInvCoef = uiXSize;

	pImgPtr = pImage;
	for(uiXCoef = 0, uiXInvCoef = uiXSize; dim8>0; dim8--, pImgPtr+=8, uiXCoef+=8, uiXInvCoef-=8){

		grayv[0] = pLUT[*pImgPtr];				grayv[1] = pLUT[*(pImgPtr+1)];
		grayv[2] = pLUT[*(pImgPtr+2)];			grayv[3] = pLUT[*(pImgPtr+3)];
		grayv[4] = pLUT[*(pImgPtr+4)];			grayv[5] = pLUT[*(pImgPtr+5)];
		grayv[6] = pLUT[*(pImgPtr+6)];			grayv[7] = pLUT[*(pImgPtr+7)];

		grayv11[0] = pulMapLU[*grayv];				grayv11[1] = pulMapLU[*(grayv+1)];
		grayv11[2] = pulMapLU[*(grayv+2)];			grayv11[3] = pulMapLU[*(grayv+3)];
		grayv11[4] = pulMapLU[*(grayv+4)];			grayv11[5] = pulMapLU[*(grayv+5)];
		grayv11[6] = pulMapLU[*(grayv+6)];			grayv11[7] = pulMapLU[*(grayv+7)];

		grayv12[0] = pulMapRU[*grayv];				grayv12[1] = pulMapRU[*(grayv+1)];
		grayv12[2] = pulMapRU[*(grayv+2)];		grayv12[3] = pulMapRU[*(grayv+3)];
		grayv12[4] = pulMapRU[*(grayv+4)];		grayv12[5] = pulMapRU[*(grayv+5)];
		grayv12[6] = pulMapRU[*(grayv+6)];		grayv12[7] = pulMapRU[*(grayv+7)];

		grayv21[0] = pulMapLB[*grayv];				grayv21[1] = pulMapLB[*(grayv+1)];
		grayv21[2] = pulMapLB[*(grayv+2)];			grayv21[3] = pulMapLB[*(grayv+3)];
		grayv21[4] = pulMapLB[*(grayv+4)];			grayv21[5] = pulMapLB[*(grayv+5)];
		grayv21[6] = pulMapLB[*(grayv+6)];			grayv21[7] = pulMapLB[*(grayv+7)];

		grayv22[0] = pulMapRB[*grayv];				grayv22[1] = pulMapRB[*(grayv+1)];
		grayv22[2] = pulMapRB[*(grayv+2)];			grayv22[3] = pulMapRB[*(grayv+3)];
		grayv22[4] = pulMapRB[*(grayv+4)];			grayv22[5] = pulMapRB[*(grayv+5)];
		grayv22[6] = pulMapRB[*(grayv+6)];			grayv22[7] = pulMapRB[*(grayv+7)];

		v11 = vld1_u8(grayv11);			v12 = vld1_u8(grayv12);
		v21 = vld1_u8(grayv21);			v22 = vld1_u8(grayv22);

		uint16x8_t v11_16 = vmovl_u8(v11);
		uint16x8_t v12_16 = vmovl_u8(v12);
		uint16x8_t v21_16 = vmovl_u8(v21);
		uint16x8_t v22_16 = vmovl_u8(v22);

		uint16x4_t v_16_low = vget_low_u16(v11_16);
		uint16x4_t v_16_high = vget_high_u16(v11_16);
		uint32x4_t v11_32_low = vmovl_u16(v_16_low);
		uint32x4_t v11_32_high = vmovl_u16(v_16_high);

		v_16_low = vget_low_u16(v12_16);
		v_16_high = vget_high_u16(v12_16);
		uint32x4_t v12_32_low = vmovl_u16(v_16_low);
		uint32x4_t v12_32_high = vmovl_u16(v_16_high);

		v_16_low = vget_low_u16(v21_16);
		v_16_high = vget_high_u16(v21_16);
		uint32x4_t v21_32_low = vmovl_u16(v_16_low);
		uint32x4_t v21_32_high = vmovl_u16(v_16_high);

		v_16_low = vget_low_u16(v22_16);
		v_16_high = vget_high_u16(v22_16);
		uint32x4_t v22_32_low = vmovl_u16(v_16_low);
		uint32x4_t v22_32_high = vmovl_u16(v_16_high);

		coefX_l = vld1q_u32(pCoefXTbl+uiXCoef);						coefX_h = vld1q_u32(pCoefXTbl+uiXCoef+4);
		coefX_inv_l = vld1q_u32(pCoefInvXTbl+uiXCoef);			coefX_inv_h = vld1q_u32(pCoefInvXTbl+uiXCoef+4);

		uint32x4_t tmp1,tmp2,tmp3,tmp4,tmp5,tmp;
		uint16x4_t result_16_low, result_16_high;
		//for low 4 numbers
		tmp1 = vmulq_u32(v11_32_low,coefX_inv_l);
		tmp2 = vmulq_u32(v12_32_low, coefX_l);
		tmp3 = vaddq_u32(tmp1, tmp2);
		tmp4 = vmulq_n_u32(tmp3, oneY_32);

		tmp1 = vmulq_u32(v21_32_low, coefX_inv_l);
		tmp2 = vmulq_u32(v22_32_low, coefX_l);
		tmp3 = vaddq_u32(tmp1, tmp2);
		tmp5 = vmulq_n_u32(tmp3, intY_32);

		tmp = vaddq_u32(tmp4, tmp5);
		result_16_low = vshrn_n_u32(tmp,16); //shift right 16 bytes
		result_16_low = vrshr_n_u16(result_16_low,8); //shift right 8 bytes, totally 24 bytes

		//for high 4 numbers
		tmp1 = vmulq_u32(v11_32_high,coefX_inv_l);
		tmp2 = vmulq_u32(v12_32_high, coefX_l);
		tmp3 = vaddq_u32(tmp1, tmp2);
		tmp4 = vmulq_n_u32(tmp3, oneY_32);

		tmp1 = vmulq_u32(v21_32_high, coefX_inv_l);
		tmp2 = vmulq_u32(v22_32_high, coefX_l);
		tmp3 = vaddq_u32(tmp1, tmp2);
		tmp5 = vmulq_n_u32(tmp3, intY_32);

		tmp = vaddq_u32(tmp4, tmp5);
		result_16_high = vshrn_n_u32(tmp,16);  //shift right 16 bytes
		result_16_high = vrshr_n_u16(result_16_high,8);  //shift right 8 bytes, totally 24 bytes

		uint16x8_t result_16 = vcombine_u16(result_16_low,result_16_high);
		uint8x8_t result_8 = vqmovn_u16(result_16);
		vst1_u8(pImgPtr, result_8);
	}

	for(; left8>0; left8--, uiXCoef++, uiXInvCoef--){

		GreyValue = (unsigned char)pLUT[(unsigned char)(*pImgPtr)];         /* get histogram bin value */
		*pImgPtr++ = (kz_pixel_t ) ((uiYInvCoef * (uiXInvCoef*pulMapLU[GreyValue] + uiXCoef * pulMapRU[GreyValue])
								  + uiYCoef * (uiXInvCoef * pulMapLB[GreyValue] + uiXCoef * pulMapRB[GreyValue])) / uiNum);
	}
}

void Interpolate_Neon(kz_pixel_t * pImage, int uiXRes, unsigned char * pulMapLU,
                  unsigned char * pulMapRU, unsigned char * pulMapLB,  unsigned char * pulMapRB,
                  unsigned int uiXSize, unsigned int uiYSize, kz_pixel_t* pLUT)
{
    const unsigned int uiIncr = uiXRes;//-uiXSize; /* Pointer increment after processing row */
    kz_pixel_t GreyValue; unsigned int uiNum = uiXSize*uiYSize; /* Normalization factor */
    unsigned char *pImgPtr;

    unsigned int uiXCoef, uiYCoef, uiXInvCoef, uiYInvCoef, uiShift = 0;
    unsigned int *pCoefXTbl,  *pCoefInvXTbl;
    int k;
    pCoefXTbl = (unsigned int*)malloc(uiXSize*2*sizeof(unsigned int));
    pCoefInvXTbl = pCoefXTbl + uiXSize;
    for(k=0; k<uiXSize; k++){
    	pCoefXTbl[k] = k*4096/uiXSize;
    	pCoefInvXTbl[k] = 4096 - pCoefXTbl[k];
    }

    for (uiYCoef = 0, uiYInvCoef = uiYSize; uiYCoef < uiYSize;  uiYCoef++, uiYInvCoef--,pImage+=uiIncr)
    {
    	interp_neon(pImage, pCoefXTbl, pCoefInvXTbl, uiYCoef, uiXSize, uiYSize, uiNum, pulMapLU, pulMapRU, pulMapLB,  pulMapRB, pLUT);
     }

    free(pCoefXTbl);
}

int CLAHE_enh_ompU8(kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
			unsigned int startX, unsigned int startY, unsigned int uiWidth, unsigned int uiHeight,
           kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
           unsigned int uiNrBins, float fCliplimit)
{
    unsigned int uiX, uiY;        /* counters */
    unsigned int uiXSize, uiYSize, uiSubX, uiSubY; /* size of context. reg. and subimages */
    unsigned int uiXL, uiXR, uiYU, uiYB;  /* auxiliary variables interpolation routine */
    unsigned int ulClipLimit, ulNrPixels;/* clip limit and region pixel count */
    kz_pixel_t* pImPointer, * pImPtr[MAX_BLOCKS_NUM];        /* pointer to image */
    kz_pixel_t aLUT[uiNR_OF_GREY];      /* lookup table used for scaling of input image */
    unsigned int*pulMapArray; /* pointer to histogram and mappings*/
    unsigned char*pulU8Array; /* pointer to histogram and mappings*/
    unsigned char* pulLU[MAX_BLOCKS_NUM], *pulLB[MAX_BLOCKS_NUM], *pulRU[MAX_BLOCKS_NUM], *pulRB[MAX_BLOCKS_NUM]; /* auxiliary pointers interpolation */
    unsigned int* pulHist[MAX_BLOCKS_NUM];
    unsigned char* pulU8Hist[MAX_BLOCKS_NUM];
	unsigned int uiSubW[MAX_BLOCKS_NUM], uiSubH[MAX_BLOCKS_NUM];
    int k,	calIdx = 0;

    if (uiNrX > uiMAX_REG_X) return -1;     /* # of regions x-direction too large */
    if (uiNrY > uiMAX_REG_Y) return -2;     /* # of regions y-direction too large */

    if (uiWidth % uiNrX) return -3;      /* x-resolution no multiple of uiNrX */
    if (uiHeight % uiNrY) return -4;      /* y-resolution no multiple of uiNrY */

    if (Max >= uiNR_OF_GREY) return -5;     /* maximum too large */
    if (Min >= Max) return -6;         /* minimum equal or larger than maximum */
    if (uiNrX < 2 || uiNrY < 2) return -7;/* at least 4 contextual regions required */
    if (fCliplimit == 1.0) return 0;      /* is OK, immediately returns original image. */
    if (uiNrBins == 0) uiNrBins = 128;    /* default value when not specified */

    if(uiNrX*uiNrY > MAX_BLOCKS_NUM)	return -9;	/* blocks number are too many*/

    pulMapArray=(unsigned int *)malloc(sizeof(unsigned int)*uiNrX*uiNrY*uiNrBins);
    if (pulMapArray == 0) return -8;      /* Not enough memory! (try reducing uiNrBins) */
    pulU8Array=(unsigned char *)malloc(sizeof(unsigned char)*uiNrX*uiNrY*uiNrBins);
    if (pulU8Array == 0) return -8;      /* Not enough memory! (try reducing uiNrBins) */

    uiXSize = uiWidth/uiNrX; uiYSize = uiHeight/uiNrY;  /* Actual size of contextual regions */
    ulNrPixels = (unsigned int)uiXSize * (unsigned int)uiYSize;

    if(fCliplimit > 0.0) {         /* Calculate actual cliplimit  */
        ulClipLimit = (unsigned int) (fCliplimit * (uiXSize * uiYSize) / uiNrBins);
        ulClipLimit = (ulClipLimit < 1UL) ? 1UL : ulClipLimit;
    }
    else ulClipLimit = 1UL<<14;         /* Large value, do not clip (AHE) */
    MakeLut(aLUT, Min, Max, uiNrBins);    /* Make lookup table for mapping of greyvalues */
    /* Calculate greylevel mappings for each contextual region */

    calIdx = 0;
    for (uiY = 0, pImPointer = (pImage+startY*uiXRes+startX); uiY < uiNrY; uiY++)
   {
	   for (uiX = 0; uiX < uiNrX; uiX++, pImPointer += uiXSize)
	   {
		   pulHist[calIdx] = &pulMapArray[uiNrBins * (uiY * uiNrX + uiX)];
		   pulU8Hist[calIdx] = &pulU8Array[uiNrBins * (uiY * uiNrX + uiX)];
		   pImPtr[calIdx] = pImPointer;
		   calIdx++;
	   }
	   pImPointer	+= (uiXRes - uiWidth);
	   pImPointer += (uiYSize - 1) * uiXRes;         /* skip lines, set pointer */
   }
#pragma omp parallel for
    for(k=0; k<calIdx; k++)
    {
    	MakeHistogram(pImPtr[k],uiXRes,uiXSize,uiYSize,pulHist[k],uiNrBins,aLUT);
    	ClipHistogram(pulHist[k], uiNrBins, ulClipLimit);
    	MapHistogramU8(pulHist[k], pulU8Hist[k], Min, Max, uiNrBins, ulNrPixels);
    }

    calIdx = 0;
    /* Interpolate greylevel mappings to get CLAHE image */
	for (pImPointer = (pImage+startY*uiXRes+startX), uiY = 0; uiY <= uiNrY; uiY++)
	{
		if (uiY == 0)
		{                     /* special case: top row */
			uiSubY = uiYSize >> 1;  uiYU = 0; uiYB = 0;
		}
		else
		{
			if (uiY == uiNrY) {               /* special case: bottom row */
				uiSubY = uiYSize >> 1;    uiYU = uiNrY-1;  uiYB = uiYU;
			}
			else
			{                     /* default values */
				uiSubY = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
			}
		}
		for (uiX = 0; uiX <= uiNrX; uiX++)
		{
			if (uiX == 0)
			{                 /* special case: left column */
				uiSubX = uiXSize >> 1; uiXL = 0; uiXR = 0;
			}
			else
			{
				if (uiX == uiNrX)
				{             /* special case: right column */
					uiSubX = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
				}
				else
				{                     /* default values */
					uiSubX = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
				}
			}

			pulLU[calIdx] = &pulU8Array[uiNrBins * (uiYU * uiNrX + uiXL)];
			pulRU[calIdx] = &pulU8Array[uiNrBins * (uiYU * uiNrX + uiXR)];
			pulLB[calIdx] = &pulU8Array[uiNrBins * (uiYB * uiNrX + uiXL)];
			pulRB[calIdx] = &pulU8Array[uiNrBins * (uiYB * uiNrX + uiXR)];
			uiSubW[calIdx] = uiSubX;
			uiSubH[calIdx] = uiSubY;
			pImPtr[calIdx] = pImPointer;

			pImPointer += uiSubX;             /* set pointer on next matrix */
			calIdx++;
		}
		pImPointer	+= (uiXRes - uiWidth);
		pImPointer += (uiSubY - 1) * uiXRes;
	}
#pragma omp parallel for
	for(k=0; k<calIdx; k++)
	{
		Interpolate_Neon(pImPtr[k],uiXRes,pulLU[k],pulRU[k],pulLB[k],pulRB[k],uiSubW[k],uiSubH[k],aLUT);
//		InterpolateU8(pImPtr[k],uiXRes,pulLU[k],pulRU[k],pulLB[k],pulRB[k],uiSubW[k],uiSubH[k],aLUT);
	}

    free(pulMapArray);                    /* free space for histograms */
    free(pulU8Array);
    return 0;                         /* return status OK */
}

int CLAHE_enh_U8(kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
			unsigned int startX, unsigned int startY, unsigned int uiWidth, unsigned int uiHeight,
           kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
           unsigned int uiNrBins, float fCliplimit)
{
    unsigned int uiX, uiY;        /* counters */
    unsigned int uiXSize, uiYSize, uiSubX, uiSubY; /* size of context. reg. and subimages */
    unsigned int uiXL, uiXR, uiYU, uiYB;  /* auxiliary variables interpolation routine */
    unsigned int ulClipLimit, ulNrPixels;/* clip limit and region pixel count */
    kz_pixel_t* pImPointer, * pImPtr[MAX_BLOCKS_NUM];        /* pointer to image */
    kz_pixel_t aLUT[uiNR_OF_GREY];      /* lookup table used for scaling of input image */
    unsigned int*pulMapArray; /* pointer to histogram and mappings*/
    unsigned char*pulU8Array; /* pointer to histogram and mappings*/
    unsigned char* pulLU[MAX_BLOCKS_NUM], *pulLB[MAX_BLOCKS_NUM], *pulRU[MAX_BLOCKS_NUM], *pulRB[MAX_BLOCKS_NUM]; /* auxiliary pointers interpolation */
    unsigned int* pulHist[MAX_BLOCKS_NUM];
    unsigned char* pulU8Hist[MAX_BLOCKS_NUM];
	unsigned int uiSubW[MAX_BLOCKS_NUM], uiSubH[MAX_BLOCKS_NUM];
    int k,	calIdx = 0;

    if (uiNrX > uiMAX_REG_X) return -1;     /* # of regions x-direction too large */
    if (uiNrY > uiMAX_REG_Y) return -2;     /* # of regions y-direction too large */

    if (uiWidth % uiNrX) return -3;      /* x-resolution no multiple of uiNrX */
    if (uiHeight % uiNrY) return -4;      /* y-resolution no multiple of uiNrY */

    if (Max >= uiNR_OF_GREY) return -5;     /* maximum too large */
    if (Min >= Max) return -6;         /* minimum equal or larger than maximum */
    if (uiNrX < 2 || uiNrY < 2) return -7;/* at least 4 contextual regions required */
    if (fCliplimit == 1.0) return 0;      /* is OK, immediately returns original image. */
    if (uiNrBins == 0) uiNrBins = 128;    /* default value when not specified */

    if(uiNrX*uiNrY > MAX_BLOCKS_NUM)	return -9;	/* blocks number are too many*/

    pulMapArray=(unsigned int *)malloc(sizeof(unsigned int)*uiNrX*uiNrY*uiNrBins);
    if (pulMapArray == 0) return -8;      /* Not enough memory! (try reducing uiNrBins) */
    pulU8Array=(unsigned char *)malloc(sizeof(unsigned char)*uiNrX*uiNrY*uiNrBins);
    if (pulU8Array == 0) return -8;      /* Not enough memory! (try reducing uiNrBins) */

    uiXSize = uiWidth/uiNrX; uiYSize = uiHeight/uiNrY;  /* Actual size of contextual regions */
    ulNrPixels = (unsigned int)uiXSize * (unsigned int)uiYSize;

    if(fCliplimit > 0.0) {         /* Calculate actual cliplimit  */
        ulClipLimit = (unsigned int) (fCliplimit * (uiXSize * uiYSize) / uiNrBins);
        ulClipLimit = (ulClipLimit < 1UL) ? 1UL : ulClipLimit;
    }
    else ulClipLimit = 1UL<<14;         /* Large value, do not clip (AHE) */
    MakeLut(aLUT, Min, Max, uiNrBins);    /* Make lookup table for mapping of greyvalues */
    /* Calculate greylevel mappings for each contextual region */

    calIdx = 0;
    for (uiY = 0, pImPointer = (pImage+startY*uiXRes+startX); uiY < uiNrY; uiY++)
   {
	   for (uiX = 0; uiX < uiNrX; uiX++, pImPointer += uiXSize)
	   {
		   pulHist[calIdx] = &pulMapArray[uiNrBins * (uiY * uiNrX + uiX)];
		   pulU8Hist[calIdx] = &pulU8Array[uiNrBins * (uiY * uiNrX + uiX)];
		   pImPtr[calIdx] = pImPointer;
		   calIdx++;
	   }
	   pImPointer	+= (uiXRes - uiWidth);
	   pImPointer += (uiYSize - 1) * uiXRes;         /* skip lines, set pointer */
   }
//#pragma omp parallel for
    for(k=0; k<calIdx; k++)
    {
    	MakeHistogram(pImPtr[k],uiXRes,uiXSize,uiYSize,pulHist[k],uiNrBins,aLUT);
    	ClipHistogram(pulHist[k], uiNrBins, ulClipLimit);
    	MapHistogramU8(pulHist[k], pulU8Hist[k], Min, Max, uiNrBins, ulNrPixels);
    }

    calIdx = 0;
    /* Interpolate greylevel mappings to get CLAHE image */
	for (pImPointer = (pImage+startY*uiXRes+startX), uiY = 0; uiY <= uiNrY; uiY++)
	{
		if (uiY == 0)
		{                     /* special case: top row */
			uiSubY = uiYSize >> 1;  uiYU = 0; uiYB = 0;
		}
		else
		{
			if (uiY == uiNrY) {               /* special case: bottom row */
				uiSubY = uiYSize >> 1;    uiYU = uiNrY-1;  uiYB = uiYU;
			}
			else
			{                     /* default values */
				uiSubY = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
			}
		}
		for (uiX = 0; uiX <= uiNrX; uiX++)
		{
			if (uiX == 0)
			{                 /* special case: left column */
				uiSubX = uiXSize >> 1; uiXL = 0; uiXR = 0;
			}
			else
			{
				if (uiX == uiNrX)
				{             /* special case: right column */
					uiSubX = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
				}
				else
				{                     /* default values */
					uiSubX = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
				}
			}

			pulLU[calIdx] = &pulU8Array[uiNrBins * (uiYU * uiNrX + uiXL)];
			pulRU[calIdx] = &pulU8Array[uiNrBins * (uiYU * uiNrX + uiXR)];
			pulLB[calIdx] = &pulU8Array[uiNrBins * (uiYB * uiNrX + uiXL)];
			pulRB[calIdx] = &pulU8Array[uiNrBins * (uiYB * uiNrX + uiXR)];
			uiSubW[calIdx] = uiSubX;
			uiSubH[calIdx] = uiSubY;
			pImPtr[calIdx] = pImPointer;

			pImPointer += uiSubX;             /* set pointer on next matrix */
			calIdx++;
		}
		pImPointer	+= (uiXRes - uiWidth);
		pImPointer += (uiSubY - 1) * uiXRes;
	}
//#pragma omp parallel for
	for(k=0; k<calIdx; k++)
	{
		Interpolate_Neon(pImPtr[k],uiXRes,pulLU[k],pulRU[k],pulLB[k],pulRB[k],uiSubW[k],uiSubH[k],aLUT);
//		InterpolateU8(pImPtr[k],uiXRes,pulLU[k],pulRU[k],pulLB[k],pulRB[k],uiSubW[k],uiSubH[k],aLUT);
	}

    free(pulMapArray);                    /* free space for histograms */
    free(pulU8Array);
    return 0;                         /* return status OK */
}



