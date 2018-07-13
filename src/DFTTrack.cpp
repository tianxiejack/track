#include "PCTracker.h"
#include "malloc_align.h"
#include "DFT.h"
#include "DFTTrack.h"

static DFT_TRK_Obj		DFTTrkObj;

static void AllocCopyMat(IMG_MAT_FLOAT	src, IMG_MAT_FLOAT *pdst)
{
	int nWidth = src.width;
	int nHeight = src.height;
	int nChannels = src.channels;
	UTILS_assert(src.data != NULL);
	if(pdst->data == NULL){
		pdst->data = (float*)MallocAlign(nWidth*nHeight*nChannels*sizeof(float));
		UTILS_assert(pdst->data != NULL);
		pdst->width = src.width;
		pdst->height = src.height;
		pdst->channels = src.channels;
		pdst->step[0] = src.step[0];
		memcpy(pdst->data, src.data , nWidth*nHeight*nChannels*sizeof(float));
	}else{
		UTILS_assert(src.width == pdst->width && src.height == pdst->height && src.channels == pdst->channels	&& src.step[0] == pdst->step[0]);
		memcpy(pdst->data, src.data , nWidth*nHeight*nChannels*sizeof(float));
	}
}

static void FreeCopyMat(IMG_MAT_FLOAT *psrc)
{
	UTILS_assert(psrc !=NULL);
	if(psrc->data != NULL){
		FreeAlign(psrc->data);
		psrc->data = NULL;
	}
	memset(psrc, 0x00, sizeof(IMG_MAT_FLOAT));
}

DFT_TRACK_HANDLE		DFT_TRK_Create()
{
	int i;
	DFT_TRK_Obj	*pTrkObj = (DFT_TRK_Obj	*)&DFTTrkObj;
	memset(pTrkObj, 0x00, sizeof(DFT_TRK_Obj));

	return (DFT_TRACK_HANDLE)(pTrkObj);
}

void DFT_TRK_setParam(DFT_TRACK_HANDLE	trkHandle, DFT_Trk_Param _prama)
{
	DFT_TRK_Obj	*pTrkObj = (DFT_TRK_Obj	*)trkHandle;
	UTILS_assert(trkHandle != NULL);
	pTrkObj->interp_factor = _prama.interp_factor;
	pTrkObj->sigma = _prama.sigma;
	pTrkObj->lambda = _prama.lambda;

	pTrkObj->padding = _prama.padding;
	pTrkObj->output_sigma_factor = _prama.output_sigma_factor;
}

void DFT_TRK_init(DFT_TRACK_HANDLE	trkHandle,	IMG_MAT_FLOAT	_alphaf,	IMG_MAT_FLOAT _prob, IMG_MAT_FLOAT _tmpl)
{
	DFT_TRK_Obj	*pTrkObj = (DFT_TRK_Obj	*)trkHandle;
	UTILS_assert(trkHandle != NULL);
	if(pTrkObj->bInited)
		return;
	AllocCopyMat(_alphaf, &pTrkObj->_alphaf);
	AllocCopyMat(_prob, &pTrkObj->_prob);
	AllocCopyMat(_tmpl, &pTrkObj->_tmpl);
	pTrkObj->bInited = 1;
}

void DFT_TRK_unInit(DFT_TRACK_HANDLE	trkHandle)
{
	DFT_TRK_Obj	*pTrkObj = (DFT_TRK_Obj	*)trkHandle;
	UTILS_assert(trkHandle != NULL);
	FreeCopyMat(&pTrkObj->_alphaf);
	FreeCopyMat(&pTrkObj->_prob);
	FreeCopyMat(&pTrkObj->_tmpl);
	pTrkObj->bInited = 0;
}

void DFT_TRK_detect(DFT_TRACK_HANDLE	trkHandle, IMG_MAT_FLOAT feature, float *peak_value, PointfCR *ptPos)
{
	DFT_TRK_Obj	*pTrkObj = (DFT_TRK_Obj	*)trkHandle;
	IMG_MAT_FLOAT resCR;
	float  peakValue;
	PointfCR	 pt;
	UTILS_assert(trkHandle != NULL);
	AllocIMGMat(&resCR,  pTrkObj->size_patch[1], pTrkObj->size_patch[0], 1);
	detectCR(pTrkObj->_tmpl, feature, &resCR, pTrkObj->_alphaf, pTrkObj->size_patch, pTrkObj->_hogfeatures, pTrkObj->sigma);
	getMinMaxValue(resCR, &pt, &peakValue);
	if(peak_value != NULL)
		*peak_value = peakValue;
	if(ptPos != NULL)
		*ptPos = pt;

	FreeIMGMat(&resCR);
}

void DFT_TRK_train(DFT_TRACK_HANDLE	trkHandle, IMG_MAT_FLOAT feature,float train_interp_factor )
{
	DFT_TRK_Obj	*pTrkObj = (DFT_TRK_Obj	*)trkHandle;
	UTILS_assert(trkHandle != NULL);

	trainCR(feature, pTrkObj->_prob, &pTrkObj->_tmpl, &pTrkObj->_alphaf, train_interp_factor, pTrkObj->lambda, pTrkObj->size_patch, pTrkObj->_hogfeatures, pTrkObj->sigma);
}
