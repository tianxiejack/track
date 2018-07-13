/*
 * Enhance.h
 *
 *  Created on: 2017年11月1日
 *      Author: cr
 */

#ifndef ENHANCE_H_
#define ENHANCE_H_

#define BYTE_IMAGE
//#undef BYTE_IMAGE

#ifdef BYTE_IMAGE

typedef unsigned char kz_pixel_t;	 /* for 8 bit-per-pixel images */
#define 	uiNR_OF_GREY (256)
#define		MAX_BLOCKS_NUM			400

#else

typedef unsigned short kz_pixel_t;	 /* for 12 bit-per-pixel images (default) */
# define uiNR_OF_GREY (4096)

#endif

/******** Prototype of CLAHE function. Put this in a separate include file. *****/
int CLAHE_enh(kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
		unsigned int startX, unsigned int startY, unsigned int uiWidth, unsigned int uiHeight,
		kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
	  unsigned int uiNrBins, float fCliplimit);

int CLAHE_enh_omp (kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
	unsigned int startX, unsigned int startY, unsigned int uiWidth, unsigned int uiHeight,
	kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
	unsigned int uiNrBins, float fCliplimit);

int CLAHE_enh_ompU8(kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
			unsigned int startX, unsigned int startY, unsigned int uiWidth, unsigned int uiHeight,
           kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
           unsigned int uiNrBins, float fCliplimit);

int CLAHE_enh_U8(kz_pixel_t* pImage, unsigned int uiXRes, unsigned int uiYRes,
			unsigned int startX, unsigned int startY, unsigned int uiWidth, unsigned int uiHeight,
           kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrX, unsigned int uiNrY,
           unsigned int uiNrBins, float fCliplimit);


#endif /* ENHANCE_H_ */
