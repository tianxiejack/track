#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "PCTracker.h"
#include "malloc_align.h"
#include "DFT.h"
#include "DFTTrack.h"
#include "HogFeat.h"
#include "RectMat.h"
#include "MatMem.h"

#include "HistTrack.h"

#define OCLUSION_THRED		(0.3f)
#define RETRY_ACQ_THRED     (0.4f)
#define	  	DYNAMIC_FRAME		(10)

static IMG_MAT *getHogFeatures(HIST_TRK_Obj* pHistTrkObj, IMG_MAT *image, IMG_MAT** out)
{
	IMG_MAT **hog = &pHistTrkObj->_FeaturesMap;
	DFT_Trk_Param		*pTrkParam = &pHistTrkObj->dftTrkParam;
	pHistTrkObj->size_patch[0] = image->height/pTrkParam->cell_size - 2;
	pHistTrkObj->size_patch[1] = image->width/pTrkParam->cell_size - 2;
	pHistTrkObj->size_patch[2] = NUM_SECTOR * 3 + 4;

	if(out != NULL)
		hog = out;

	if((*hog) == NULL)
		(*hog) = matAlloc(MAT_float, pHistTrkObj->size_patch[2], pHistTrkObj->size_patch[1]*pHistTrkObj->size_patch[0], 1);
	else{
		UTILS_assert((*hog)->size == pHistTrkObj->size_patch[2]*pHistTrkObj->size_patch[1]*pHistTrkObj->size_patch[0]*sizeof(float));
		(*hog)->width = pHistTrkObj->size_patch[2];
		(*hog)->height = pHistTrkObj->size_patch[1]*pHistTrkObj->size_patch[0];
		(*hog)->step[0]= pHistTrkObj->size_patch[2];
	}
	UTILS_assert((*hog) != NULL);
	memset((*hog)->data, 0, (*hog)->size);

	GetHogFeat(image, (*hog), pTrkParam->cell_size, NUM_SECTOR);

	transpose((*hog));

	return (*hog);
}

// Initialize Hanning window. Function called only in the first frame.
static void createHanningMats(HIST_TRK_Obj* pHistTrkObj)
{
	int i, j;
	IMG_MAT* hann1t = matAlloc(MAT_float,pHistTrkObj->size_patch[1], 1, 1);
	IMG_MAT* hann2t = matAlloc(MAT_float,1,pHistTrkObj->size_patch[0], 1);
	IMG_MAT* hann3t = matAlloc(MAT_float,pHistTrkObj->size_patch[1],pHistTrkObj->size_patch[0], 1);

	for(i=0; i<hann1t->width; i++)
		hann1t->data[i] = 0.5f * (1 - cos(2.0f * 3.14159265358979323846f * i / (hann1t->width - 1)));
	for(i=0; i<hann2t->height; i++)
		hann2t->data[i] = 0.5f * (1 - cos(2.0f * 3.14159265358979323846f * i / (hann2t->height- 1)));

	for(j=0; j<hann3t->height; j++)
	{
		for(i=0; i<hann3t->width; i++)
			hann3t->data[j*hann3t->step[0]+i] = hann2t->data[j]*hann1t->data[i];
	}

	pHistTrkObj->_hann = matAlloc(MAT_float, pHistTrkObj->size_patch[0]*pHistTrkObj->size_patch[1], pHistTrkObj->size_patch[2], 1);

	for (i = 0; i < pHistTrkObj->size_patch[2]; i++) {
		for (j = 0; j<pHistTrkObj->size_patch[0]*pHistTrkObj->size_patch[1]; j++) {
			pHistTrkObj->_hann->data[pHistTrkObj->_hann->step[0]*i+j] = hann3t->data[j];
		}
	}

	matFree(hann1t);
	matFree(hann2t);
	matFree(hann3t);
}

// Obtain sub-window from image, with replication-padding and extract features
//static
static IMG_MAT* getFeatures(	HIST_TRK_Obj* pHistTrkObj, IMG_MAT *image, bool inithann,
															float scale_adjust, UTC_RECT_float roi, IMG_MAT** out, IMG_MAT *imgMap)
{
	IMG_MAT* retMat = NULL;
	IMG_MAT* subimage = NULL;
	Recti extracted_roi;
	Recti subRect;
	Sizei subSize;

	float cx = roi.x + roi.width / 2;
	float cy = roi.y + roi.height / 2;
	DFT_Trk_Param		*pTrkParam = &pHistTrkObj->dftTrkParam;
	int cell_size = pTrkParam->cell_size;

	//_beginTm = Utils_getCurTimeInMsec();

	if (inithann)
	{
		int padded_w = (int)(roi.width * pTrkParam->padding +0.5f);
		int padded_h = (int)(roi.height * pTrkParam->padding+0.5f);

		if (pTrkParam->template_size > 1) {  // Fit largest dimension to the given template size
			if (padded_w >= padded_h)  //fit to width
				pHistTrkObj->_scale = padded_w / (float) pTrkParam->template_size;
			else
				pHistTrkObj->_scale = padded_h / (float) pTrkParam->template_size;

			pHistTrkObj->_tmpl_sz.width = (int)(padded_w / pHistTrkObj->_scale);
			pHistTrkObj->_tmpl_sz.height = (int)(padded_h / pHistTrkObj->_scale);
		}
		else {  //No template size given, use ROI size
			pHistTrkObj->_tmpl_sz.width = padded_w;
			pHistTrkObj->_tmpl_sz.height = padded_h;
			pHistTrkObj->_scale = 1;
		}
		pHistTrkObj->_scaleBK = pHistTrkObj->_scale;
		// Round to cell size and also make it even
		pHistTrkObj->_tmpl_sz.width = ( ( (int)(pHistTrkObj->_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
		pHistTrkObj->_tmpl_sz.height = ( ( (int)(pHistTrkObj->_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
	}

	extracted_roi.width = (int)(scale_adjust * pHistTrkObj->_scale * pHistTrkObj->_tmpl_sz.width);
	extracted_roi.height = (int)(scale_adjust * pHistTrkObj->_scale * pHistTrkObj->_tmpl_sz.height);

	// center roi with new size
	extracted_roi.x = (int)(cx - extracted_roi.width / 2);
	extracted_roi.y = (int)(cy - extracted_roi.height / 2);

	subRect.x = extracted_roi.x;
	subRect.y = extracted_roi.y;
	subRect.width = extracted_roi.width;
	subRect.height = extracted_roi.height;
	subSize.width = pHistTrkObj->_tmpl_sz.width;
	subSize.height = pHistTrkObj->_tmpl_sz.height;

	subimage = matAlloc(image->dtype,	subSize.width, subSize.height, image->channels);
	if(subRect.x<=0 && subRect.y<=0 && subRect.x+subRect.width>= image->width && subRect.y+subRect.height>= image->height){
		ResizeMat(*image, subimage);
	}else{
		SubWindowMat(*image, subimage, subRect, subSize);
	}

	// HOG features
	retMat = getHogFeatures(pHistTrkObj, subimage, out);
	matFree(subimage);

	if (inithann) {
		createHanningMats(pHistTrkObj);
	}

	retMat = matMul(retMat, pHistTrkObj->_hann);

	return retMat;
}

// Detect object in the current frame.
//static
static PointfCR detect(HIST_TRK_Obj* pHistTrkObj, IMG_MAT templ, IMG_MAT feature, float *pPeak_value, IMG_MAT *resMat)
{
	PointfCR pt;
	IMG_MAT *res;
	float  peakValue;

	res = matAlloc(MAT_float, pHistTrkObj->size_patch[1], pHistTrkObj->size_patch[0], 1);
	detectCR(templ, feature, res, *pHistTrkObj->_alphaf, pHistTrkObj->size_patch, true, pHistTrkObj->dftTrkParam.sigma);

	getMinMaxValue(*res, &pt, &peakValue);
	if(pPeak_value != NULL)
		*pPeak_value = peakValue;
	if(resMat != NULL){
		matCopy(resMat, res);
	}

	matFree(res);

	return pt;
}

// Initialize tracker
static int init(HIST_TRK_Obj* pHistTrkObj, UTC_RECT_float roi, IMG_MAT *image)
{
	int iRet = 0;
	int i;
	IMG_MAT *fTmp = NULL;
	int step = 0;

	UTILS_assert(roi.width >= 0 && roi.height >= 0);
	pHistTrkObj->_roi = roi;

	fTmp = getFeatures(pHistTrkObj, image, 1, 1.0f, pHistTrkObj->_roi, NULL, NULL);

	pHistTrkObj->_tmpl = matAlloc(fTmp->dtype, fTmp->width, fTmp->height, fTmp->channels);
	matCopy(pHistTrkObj->_tmpl, fTmp);

	pHistTrkObj->_prob = matAlloc(MAT_float, pHistTrkObj->size_patch[1], pHistTrkObj->size_patch[0], 2);

	createGaussianPeak(*pHistTrkObj->_prob, pHistTrkObj->size_patch[0], pHistTrkObj->size_patch[1],
											pHistTrkObj->dftTrkParam.padding, pHistTrkObj->dftTrkParam.output_sigma_factor);

	pHistTrkObj->_alphaf = matAlloc(MAT_float, pHistTrkObj->size_patch[1], pHistTrkObj->size_patch[0], 2);

	for(i=0; i<HIST_SAVE_NUM; i++)
	{
		pHistTrkObj->_feature_tmpl[i] = matAlloc(MAT_float, pHistTrkObj->_tmpl->width, pHistTrkObj->_tmpl->height, pHistTrkObj->_tmpl->channels);
		pHistTrkObj->_feature_alphaf[i] = matAlloc(MAT_float, pHistTrkObj->_alphaf->width, pHistTrkObj->_alphaf->height, pHistTrkObj->_alphaf->channels);
		pHistTrkObj->_feature_peak[i] = 0.0f;
	}

	pHistTrkObj->featIndx = 0;
	pHistTrkObj->lostFrame = 0;
	pHistTrkObj->utcStatus = 1;

	// train with initial frame
	trainCR(*fTmp, *pHistTrkObj->_prob, pHistTrkObj->_tmpl, pHistTrkObj->_alphaf, 1.0, pHistTrkObj->dftTrkParam.lambda, pHistTrkObj->size_patch, true, pHistTrkObj->dftTrkParam.sigma);

	pHistTrkObj->_feature_peak[pHistTrkObj->featIndx ] = 1.0f;
	matCopy(pHistTrkObj->_feature_tmpl[pHistTrkObj->featIndx], pHistTrkObj->_tmpl);
	matCopy(pHistTrkObj->_feature_alphaf[pHistTrkObj->featIndx], pHistTrkObj->_alphaf);
	pHistTrkObj->featIndx++;
	if(pHistTrkObj->featIndx == HIST_SAVE_NUM)
		pHistTrkObj->featIndx = 0;

	return iRet;
}

static void unInit(HIST_TRK_Obj* pHistTrkObj)
{
	int i;
	matFree(pHistTrkObj->_hann);
	matFree(pHistTrkObj->_tmpl);
	matFree(pHistTrkObj->_alphaf);
	matFree(pHistTrkObj->_prob);
	matFree(pHistTrkObj->_FeaturesMap);
	matFree(pHistTrkObj->_templateMap);
	matFree(pHistTrkObj->_curImageMap);
	pHistTrkObj->_hann = NULL;
	pHistTrkObj->_tmpl = NULL;
	pHistTrkObj->_alphaf = NULL;
	pHistTrkObj->_prob = NULL;
	pHistTrkObj->_FeaturesMap = NULL;
	pHistTrkObj->_templateMap = NULL;
	pHistTrkObj->_curImageMap = NULL;

	for(i=0; i<HIST_SAVE_NUM; i++)
	{
		matFree(pHistTrkObj->_feature_tmpl[i]);
		pHistTrkObj->_feature_tmpl[i] = NULL;
		matFree(pHistTrkObj->_feature_alphaf[i]);
		pHistTrkObj->_feature_alphaf[i] = NULL;
		pHistTrkObj->_feature_peak[i] = 0.0f;
	}
}

// Constructor
static int create(HIST_TRK_Obj* pHistTrkObj, bool multiscale, bool fixed_window)
{
	int iRet = 0;
	DFT_Trk_Param		*pTrkParam = &pHistTrkObj->dftTrkParam;
	// Parameters equal in all cases
	pTrkParam->lambda = 0.0001f;
	pTrkParam->padding = 2.5f;
	pTrkParam->output_sigma_factor = 0.125f;
	//padding = 3.0;
	//output_sigma_factor = 0.1042;

	// VOT
	pTrkParam->interp_factor = 0.012f;
	pTrkParam->sigma = 0.6f;
	// TPAMI
	//interp_factor = 0.02;
	//sigma = 0.5;
	pTrkParam->cell_size = 4;

	if (multiscale) { // multiscale
		pTrkParam->template_size = 96;
		pTrkParam->scale_step = 1.05f;
		pTrkParam->scale_weight = 0.95f;
		if (!fixed_window) {
			//printf("Multiscale does not support non-fixed window.\n");
			fixed_window = true;
		}
	}
	else if (fixed_window) {  // fit correction without multiscale
		pTrkParam->template_size = 96;
		//template_size = 100;
		pHistTrkObj->dftTrkParam.scale_step = 1;
	}
	else {
		pTrkParam->template_size = 1;
		pTrkParam->scale_step = 1;
	}

	return iRet;
}

static void destroy(HIST_TRK_Obj* pHistTrkObj)
{
	unInit(pHistTrkObj);
}

static void initparam(HIST_TRK_Obj* pHistTrkObj)
{
	DFT_Trk_Param		*pTrkParam = &pHistTrkObj->dftTrkParam;
	memset(pHistTrkObj, 0, sizeof(HIST_TRK_Obj));

	pHistTrkObj->gDynamicParam.occlusion_thred = OCLUSION_THRED;
	pHistTrkObj->gDynamicParam.retry_acq_thred = RETRY_ACQ_THRED;
	pHistTrkObj-> adaptive_acq_thred = RETRY_ACQ_THRED;
	pHistTrkObj->gIntevlFrame = DYNAMIC_FRAME;

	pHistTrkObj->featIndx = 0;
	pHistTrkObj->lostFrame = 0;
	pHistTrkObj->utcStatus = 0;

	pHistTrkObj->opt_peak_value = 0.0f;
	pHistTrkObj->opt_peak_value_search = 0.0f;
	pHistTrkObj->opt_apceVal_search = 0.0f;
}

// Update position based on the new frame
static UTC_RECT_float update(HIST_TRK_Obj* pHistTrkObj, IMG_MAT *image)
{
	float cx, cy;
	float peak_value;
	PointfCR res;
	PointfCR centerPt;
	IMG_MAT *features;
	DFT_Trk_Param		*pTrkParam = &pHistTrkObj->dftTrkParam;
	int cell_size = pTrkParam->cell_size;

	if (pHistTrkObj->_roi.x + pHistTrkObj->_roi.width <= 0) pHistTrkObj->_roi.x = -pHistTrkObj->_roi.width + 1;
	if (pHistTrkObj->_roi.y + pHistTrkObj->_roi.height <= 0) pHistTrkObj->_roi.y = -pHistTrkObj->_roi.height + 1;
	if (pHistTrkObj->_roi.x >= image->width - 1) pHistTrkObj->_roi.x = (float)image->width - 2;
	if (pHistTrkObj->_roi.y >= image->height - 1) pHistTrkObj->_roi.y = (float)image->height - 2;

	cx = pHistTrkObj->_roi.x + pHistTrkObj->_roi.width / 2.0f;
	cy = pHistTrkObj->_roi.y + pHistTrkObj->_roi.height / 2.0f;

	features = getFeatures(pHistTrkObj, image, 0, 1.0f, pHistTrkObj->_roi, NULL, NULL);
	res = detect(pHistTrkObj, *pHistTrkObj->_tmpl, *features, &peak_value, NULL);

	if (pTrkParam->scale_step != 1) {
		// Test at a smaller _scale
		float new_peak_value;
		PointfCR new_res;
		features = getFeatures(pHistTrkObj, image, 0, 1.0f/ pTrkParam->scale_step, pHistTrkObj->_roi, NULL, NULL);
		new_res = detect(pHistTrkObj, *pHistTrkObj->_tmpl, *features, &new_peak_value, NULL);

		if (pTrkParam->scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			pHistTrkObj->_scale /= pTrkParam->scale_step;
			pHistTrkObj->_roi.width /= pTrkParam->scale_step;
			pHistTrkObj->_roi.height /= pTrkParam->scale_step;
		}

		// Test at a bigger _scale
		features = getFeatures(pHistTrkObj, image, 0, pTrkParam->scale_step, pHistTrkObj->_roi, NULL, NULL);
		new_res = detect(pHistTrkObj, *pHistTrkObj->_tmpl, *features, &new_peak_value,NULL);

		if (pTrkParam->scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			pHistTrkObj->_scale *= pTrkParam->scale_step;
			pHistTrkObj->_roi.width *= pTrkParam->scale_step;
			pHistTrkObj->_roi.height *= pTrkParam->scale_step;
		}
	}
	pHistTrkObj->opt_peak_value = peak_value;
	if(pHistTrkObj->opt_peak_value <pHistTrkObj->gDynamicParam.occlusion_thred*0.68){
		res.x = res.y = 0.0;
	}

	// Adjust by cell size and _scale
	pHistTrkObj->_roi.x = cx - pHistTrkObj->_roi.width / 2.0f + ((float) res.x * cell_size * pHistTrkObj->_scale);
	pHistTrkObj->_roi.y = cy - pHistTrkObj->_roi.height / 2.0f + ((float) res.y * cell_size * pHistTrkObj->_scale);

	if (pHistTrkObj->_roi.x >= image->width - 1) pHistTrkObj->_roi.x = (float)image->width - 1;
	if (pHistTrkObj->_roi.y >= image->height - 1) pHistTrkObj->_roi.y = (float)image->height - 1;
	if (pHistTrkObj->_roi.x + pHistTrkObj->_roi.width <= 0) pHistTrkObj->_roi.x = -pHistTrkObj->_roi.width + 2;
	if (pHistTrkObj->_roi.y + pHistTrkObj->_roi.height <= 0) pHistTrkObj->_roi.y = -pHistTrkObj->_roi.height + 2;
	UTILS_assert(pHistTrkObj->_roi.width >= 0 && pHistTrkObj->_roi.height >= 0);

	features = getFeatures(pHistTrkObj, image, 0, 1.0f, pHistTrkObj->_roi, NULL, NULL);

	trainCR(	*features, *pHistTrkObj->_prob, pHistTrkObj->_tmpl, pHistTrkObj->_alphaf,
						pTrkParam->interp_factor, pTrkParam->lambda, pHistTrkObj->size_patch, true, pTrkParam->sigma);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	pHistTrkObj->_scaleBK = pHistTrkObj->_scale;
	pHistTrkObj->_roiBK = pHistTrkObj->_roi;

	if(pHistTrkObj->opt_peak_value < pHistTrkObj->gDynamicParam.occlusion_thred)
	{
		pHistTrkObj->lostFrame++;
		pHistTrkObj->utcStatus = 0;
	}
	else
	{
		pHistTrkObj->_feature_peak[pHistTrkObj->featIndx ] = pHistTrkObj->opt_peak_value;
		matCopy(pHistTrkObj->_feature_tmpl[pHistTrkObj->featIndx], pHistTrkObj->_tmpl);
		matCopy(pHistTrkObj->_feature_alphaf[pHistTrkObj->featIndx], pHistTrkObj->_alphaf);
		pHistTrkObj->featIndx++;
		if(pHistTrkObj->featIndx == HIST_SAVE_NUM)
			pHistTrkObj->featIndx = 0;

		pHistTrkObj->lostFrame = 0;
		pHistTrkObj->utcStatus  = 1;
	}
	return pHistTrkObj->_roi;
}

// Update position based on the new frame
static UTC_RECT_float update_Hist(HIST_TRK_Obj* pHistTrkObj,  IMG_MAT *image, BACK_ENUM_TYPE	backType, bool bUpdata)
{
	float cx, cy;
	float peak_value;
	PointfCR res;
	IMG_MAT *features;
	DFT_Trk_Param		*pTrkParam = &pHistTrkObj->dftTrkParam;
	int cell_size = pTrkParam->cell_size;
	float	occlusionThed = (backType==BACK_TRK_TYPE)?pHistTrkObj->gDynamicParam.occlusion_thred:pHistTrkObj->gDynamicParam.retry_acq_thred;

	if (pHistTrkObj->_roi.x + pHistTrkObj->_roi.width <= 0) pHistTrkObj->_roi.x = -pHistTrkObj->_roi.width + 1;
	if (pHistTrkObj->_roi.y + pHistTrkObj->_roi.height <= 0) pHistTrkObj->_roi.y = -pHistTrkObj->_roi.height + 1;
	if (pHistTrkObj->_roi.x >= image->width - 1) pHistTrkObj->_roi.x = (float)image->width - 2;
	if (pHistTrkObj->_roi.y >= image->height - 1) pHistTrkObj->_roi.y = (float)image->height - 2;

	cx = pHistTrkObj->_roi.x + pHistTrkObj->_roi.width / 2.0f;
	cy = pHistTrkObj->_roi.y + pHistTrkObj->_roi.height / 2.0f;

	features = getFeatures(pHistTrkObj, image, 0, 1.0f, pHistTrkObj->_roi, NULL, NULL);
	res = detect(pHistTrkObj, *pHistTrkObj->_tmpl, *features,  &peak_value, NULL);

	pHistTrkObj->opt_peak_value = peak_value;
	if(pHistTrkObj->opt_peak_value <pHistTrkObj->gDynamicParam.occlusion_thred*0.68){
		res.x = res.y = 0.0;
	}

	// Adjust by cell size and _scale
	pHistTrkObj->_roi.x = cx - pHistTrkObj->_roi.width / 2.0f + ((float) res.x * cell_size * pHistTrkObj->_scale);
	pHistTrkObj->_roi.y = cy - pHistTrkObj->_roi.height / 2.0f + ((float) res.y * cell_size * pHistTrkObj->_scale);

	if (pHistTrkObj->_roi.x >= image->width - 1) pHistTrkObj->_roi.x = (float)image->width - 1;
	if (pHistTrkObj->_roi.y >= image->height - 1) pHistTrkObj->_roi.y = (float)image->height - 1;
	if (pHistTrkObj->_roi.x + pHistTrkObj->_roi.width <= 0) pHistTrkObj->_roi.x = -pHistTrkObj->_roi.width + 2;
	if (pHistTrkObj->_roi.y + pHistTrkObj->_roi.height <= 0) pHistTrkObj->_roi.y = -pHistTrkObj->_roi.height + 2;
	UTILS_assert(pHistTrkObj->_roi.width >= 0 && pHistTrkObj->_roi.height >= 0);

	pHistTrkObj->_scaleBK = pHistTrkObj->_scale;
	pHistTrkObj->_roiBK = pHistTrkObj->_roi;
//	printf("%s:opt=%0.2f occlusionThed=%0.2f\n",__func__, pHistTrkObj->opt_peak_value, occlusionThed);

//	if(pDftTrkObj->opt_peak_value < pDftTrkObj->gDynamicParam.occlusion_thred)
	if(pHistTrkObj->opt_peak_value < occlusionThed)
	{
		pHistTrkObj->lostFrame++;
		pHistTrkObj->utcStatus = 0;
	}
	else
	{
		if(bUpdata){
			trainCR(	*features, *pHistTrkObj->_prob, pHistTrkObj->_tmpl, pHistTrkObj->_alphaf,
						pTrkParam->interp_factor, pTrkParam->lambda, pHistTrkObj->size_patch, true, pTrkParam->sigma);
		}
		pHistTrkObj->lostFrame = 0;
		pHistTrkObj->utcStatus  = 1;
	}

	return pHistTrkObj->_roi;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

HIST_TRACK_HANDLE CreateHistTrk(bool multiscale, bool fixed_window)
{
	HIST_TRK_Obj	*pHistTrkObj = (HIST_TRK_Obj*)malloc(sizeof(HIST_TRK_Obj));
	UTILS_assert(pHistTrkObj != NULL);

	initparam(pHistTrkObj);

	create(pHistTrkObj, multiscale, fixed_window);

	pHistTrkObj->m_backHdl = CreatBackHandle();

	pHistTrkObj->UtcTrkObj.m_bInited = true;
	pHistTrkObj->UtcTrkObj.axisX = 1920/2;
	pHistTrkObj->UtcTrkObj.axisY = 1080/2;

	return (HIST_TRACK_HANDLE)pHistTrkObj;
}

void DestroySceneTrk(HIST_TRACK_HANDLE handle)
{
	HIST_TRK_Obj	*pTrkObj  = (HIST_TRK_Obj*)handle;

	destroy(pTrkObj);

	DestroyBackHandle(pTrkObj->m_backHdl);
	pTrkObj->m_backHdl = NULL;

	memset(pTrkObj, 0, sizeof(HIST_TRK_Obj));
	if(pTrkObj != NULL)
		free(pTrkObj);
}

UTC_RECT_float HistTrkAcq(HIST_TRACK_HANDLE pHistTrkObj, IMG_MAT frame, UTC_ACQ_param inputParam)
{
	UTC_RECT_float rcInit;

	pHistTrkObj->UtcTrkObj.axisX = inputParam.axisX;
	pHistTrkObj->UtcTrkObj.axisY = inputParam.axisY;
	rcInit.x = (float)inputParam.rcWin.x;
	rcInit.y = (float)inputParam.rcWin.y;
	rcInit.width = (float)inputParam.rcWin.width;
	rcInit.height = (float)inputParam.rcWin.height;

	UTILS_assert(frame.width == frame.step[0]);

	unInit(pHistTrkObj);

	init(pHistTrkObj, rcInit, &frame);

	return rcInit;
}

UTC_RECT_float HistTrkProc(HIST_TRACK_HANDLE pHistTrkObj, IMG_MAT frame, int *pRtnStat, bool bUpdata)
{
	UTC_RECT_float rcResult;
	UTILS_assert(frame.width == frame.step[0]);

//	rcResult = update(pHistTrkObj, &frame);
	rcResult = update_Hist(pHistTrkObj, &frame, BACK_TRK_TYPE, bUpdata);

	pHistTrkObj->_offsetX = floor((rcResult.x+rcResult.width/2.f) - pHistTrkObj->UtcTrkObj.axisX+0.5);
	pHistTrkObj->_offsetY = floor((rcResult.y+rcResult.height/2.f) - pHistTrkObj->UtcTrkObj.axisY+0.5);

	if(pRtnStat != NULL){
		if(pHistTrkObj->utcStatus == 0)	// 0-lost 1-trk
			*pRtnStat = 2;	// assi
		else
			*pRtnStat = 1;	// trk
	}

	return rcResult;
}

int	 BackTrackStatus(HIST_TRACK_HANDLE pHistTrkObj, SCENE_MV_HANDLE pSceneMVObj, IMG_MAT frame, UTCTRACK_OBJ UtcTrkObj,
		float opt,	UTC_RECT_float roi, int utcStatus)
{
	UTC_ACQ_param	inputParam;
	inputParam.axisX = UtcTrkObj.axisX;
	inputParam.axisY = UtcTrkObj.axisY;
	inputParam.rcWin.x= (int)roi.x;
	inputParam.rcWin.y= (int)roi.y;
	inputParam.rcWin.width= (int)roi.width;
	inputParam.rcWin.height= (int)roi.height;

	HistTrkAcq(pHistTrkObj, frame, inputParam);

	BACK_OBJ *pBackObj = (BACK_OBJ*)pHistTrkObj->m_backHdl;
	initBackObj(pBackObj, roi, &frame);

	BackTrkProc(pHistTrkObj, frame, opt, roi, utcStatus, pSceneMVObj->m_trajParam.m_histDelta, NULL);

	return pBackObj->backStatus;
}




