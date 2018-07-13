#include <math.h>
#include "SaliencyProc.h"
#include "MVDetector.h"
#include "MatMem.h"
#include "PCTracker.h"
#include "UtcTrack.h"
#include "DFT.h"
#include "HogFeat.h"
#include "RectMat.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "Enhance.h"
#include "SalientSR.h"
#include "ConnectRegion.h"
#include "SceneMV.h"
#include "HistTrack.h"
#include "osa.h"
#include "osa_mutex.h"
#include "osa_tsk.h"
#include "osa_sem.h"
#include "CKalmanTrk.h"

/************************************************************************
*
*
*
*/
#define SAVE_NUM	(30)
#define OCLUSION_THRED		(0.3f)
#define RETRY_ACQ_THRED     (0.4f)
static UTC_RECT_float _roi, _roiBK;
static float interp_factor; // linear interpolation factor for adaptation
static float sigma; // gaussian kernel bandwidth
static float lambda; // regularization
static int cell_size; // HOG cell size
//static int cell_sizeQ; // cell size^2, to avoid repeated operations
static float padding; // extra area surrounding the target
static float output_sigma_factor; // bandwidth of gaussian target
static int template_size; // template size
static float scale_step; // scale step for multi-scale estimation
static float scale_weight;  // to downweight detection scores of other scales for added stability

static UTC_SIZE _tmpl_sz;
static float _scale, _scaleBK,_maxpeak;
static int size_patch[3];

static IMG_MAT *_FeaturesMap = NULL;
static IMG_MAT *_tmpl = NULL;
static IMG_MAT *_hann = NULL;
static IMG_MAT *_alphaf = NULL;
static IMG_MAT *_prob = NULL;
static IMG_MAT *_feature_tmpl[SAVE_NUM] = {NULL,};
static IMG_MAT *_feature_alphaf[SAVE_NUM] = {NULL,};
static float _feature_peak[SAVE_NUM] = {0.0f,};

static int	 featIndx = 0;
static int   lostFrame = 0;
static int   utcStatus = 0;
static int   backStatus = 0;

static float opt_peak_value = 0.0f;
static float opt_peak_value_search = 0.0;
static UTC_RECT_float opt_roi_search;

static UTC_DYN_PARAM	gDynamicParam;
static SEARCH_MODE_TYPE	gSerchMode = SEARCH_MODE_ALL;
static int intervalFrame = 0, gIntervalFrame = 10;
static float adaptive_acq_thred = RETRY_ACQ_THRED;
static int _bBigSearch = false;

static UInt32 _beginTm = 0, _curTm = 0;

static int _platType = tPLT_TST;
static int	_bsType = BoreSight_Mid;
static float _blendValue = 16;
static int _offsetX = 0, _offsetY = 0;

static	int	min_bias_pix	= 200;
static	int	max_bias_pix = 600;

static	int first_print = 0;
static  TRK_SECH_RESTRAINT	resTrackObj={
	.res_distance = 80,
	.res_area = 5000,
};

static int _roiMaxWidth = 0xFFFFFFF;

static int _roiMinWidth = 4;

typedef enum{
	_work_mode_trk = 0x01,
	_work_mode_serch = 0x02,
}WORK_MODE;
static int _workMode = _work_mode_trk;

static bool _bEnhROI = false;//true;
static bool _bBlurFilter = false;//true;
static float _fCliplimit = 4.0;
static bool _bPrintTS = false;
static bool _bPredict = false;
static UTC_RECT_float _trkSchAcqRc, _trkSchLstRc;
static bool	_bTrkSchLost = false;
static PointICR	_iMvPixel={
	.x = 20,
	.y = 10,
};

static PointICR	_iMvPixel2={
	.x = 30,
	.y = 20,
};

static PointICR	 _iSegPixel={
	.x = 600,
	.y = 450,
};

static int 	_salientThred = 40;
static int 	_salientScatter = 10;
static int   _largeScaler = 256;
static int 	_midScaler	= 128;
static int   _smallScaler = 64;
static int 	_bmultScaler = 1;
static UTC_SIZE	_minAcqSize={
		.width = 8,
		.height = 8,
};
static float	_acqRatio = 1.0;
static bool	_bDynamicRatio = false;

static bool	_bCalSceneMV = false;
static bool	_bBackTrack = false;
static bool	_bStartBackTrk = false;
static bool	_bSceneMVRecord = false;
static UInt32 startFrames = 0;
static UInt32  continueCST = 0;
static PointfCR	 speedCST={
		.x = 0.f,
		.y = 0.f,
};

#define	 TRK_POS_NUM			15
static PointfCR	 _track_pos[TRK_POS_NUM];
static int trkPosNmu=0;
static UTCTRACK_OBJ *pTrkObj = NULL;
static bool _bAveTrkPos = false;
#define		DETECTOR_HISTORY_FRAMES	50
static float	_detectfTau = 0.5;
static int		_detectBuildFrms = 500;
static int		_LostFrmThred = 30;
static float	_histMvThred = 1.0;
static int	_detectorHistFrms = 30;
static float	_stillThred = 0.1;
static int 	_stillFrms = 50;

typedef enum{
	IDLE_MODEL_STATUS 		= 0x00,
	TRACK_MODEL_STATUS	,
	SEARCH_MODEL_STATUS,
	BACK_MODEL_STATUS,
};

static int	_bTrackModel = IDLE_MODEL_STATUS;
static int	_nThredFrms = 30;
static float	_fRatioFrms = 0.25;
static bool	_bKalmanFilter = false;
static PointfCR		_KalmanMVThred={
		.x = 3.0,
		.y = 2.0,
};
static PointfCR		_KalmanStillThred={
		.x = 0.5,
		.y = 0.3,
};
static float _slopeThred = 0.08;
static float	_KalmanHistThred = 2.5;
static float	_KalmanCoefQ = 0.00001;
static float	_KalmanCoefR = 0.0025;

static float	_trkAngleThred = 60.0;
static bool	_bHighOpt = false;
static bool	_bEnhScene = false;
static unsigned int		_nrxScene = 2;
static unsigned int		_nryScene = 2;
static int 		_sceneFrmCount = 0;
static int 		_sceneFrmGap = 1;
static int 		_sceneFrmThred = 5;
static int		_sceneFrmSum = 0;

static IMG_MAT *_tskMap = NULL;

static IMG_MAT *_origtmpl = NULL;
static IMG_MAT *_origalphaf = NULL;

static IMG_MAT *_peakMat = NULL;
static UTC_RECT_float  _peakRoi;
static PointfCR		_peakRes;
static float	_peakMvThred = 1.0;
static UInt32  continuePeak = 0;

static PointfCR	 _servoSpeed={
		.x = 0.f,
		.y = 0.f,
};

typedef enum{
	IDLE_SIM_STATUS 		= 0x00,
	TRACK_SIM_STATUS	,
	OVERLAP_SIM_STATUS,
	SEARCH_SIM_STATUS,
};

static bool	_bSimTrack = false;
static int	_bSimStatus = IDLE_SIM_STATUS;

typedef struct{
	float opt_peak_value;// = 0.0f;
	UTC_RECT_float _roi;
	int   utcStatus;// = 0;

	bool	bsim;//is have similar target in the trajectory
	UTC_RECT_float		simRect;
}SUB_TRK_PARAM;
/********************************************************/

static UTCTRACK_OBJ UtcTrk0bj;

static CON_REG_HANDLE gConRegHdl = NULL;

static	SCENE_MV_HANDLE		gSceneMVHdl = NULL;

static HIST_TRACK_HANDLE	gHistTrkHdl = NULL;

static CMoveDetector	gMVDetectObj;

static CKalmanTracker	gKalmanTrkObj;

static SUB_TRK_PARAM	gSubTrkParam;

 /********************************************************/
static void unInit();

static UInt32 dbg_tmStat;
static UInt32 dbg_getCurTimeInUsec()
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

static bool _overlapRoi(cv::Rect rec1,	cv::Rect rec2, cv::Rect *roi)
{
	cv::Point2d tl1, tl2;
	cv::Size sz1,sz2;
	int x_tl, y_tl, x_br, y_br;
	tl1.x	= rec1.x;	tl1.y	= rec1.y;
	tl2.x	= rec2.x;	tl2.y	= rec2.y;
	sz1.width	= rec1.width;	sz1.height	= rec1.height;
	sz2.width	= rec2.width;	sz2.height	= rec2.height;

	x_tl = MAX(tl1.x, tl2.x);
	y_tl = MAX(tl1.y, tl2.y);
	x_br = MIN(tl1.x + sz1.width, tl2.x + sz2.width);
	y_br = MIN(tl1.y + sz1.height, tl2.y + sz2.height);
	if (x_tl < x_br && y_tl < y_br)
	{
		roi->x = x_tl;	roi->y = y_tl;	roi->width = x_br - x_tl;	roi->height = y_br - y_tl;
		return true;
	}
	return false;
}

static float _getAngle(PointfCR hist_delta)
{
	float hist_sqrt = sqrt(hist_delta.x*hist_delta.x + hist_delta.y*hist_delta.y);
	float hist_angle = (float)asin(hist_delta.y/ hist_sqrt);//(-pi/2,pi/2)
	hist_angle *= (float)180.f/CR_PI;
	if(hist_delta.x < 0){
		hist_angle = 180.0 - hist_angle;
	}else{
		if(hist_delta.y <0){
			hist_angle += 360.0;
		}
	}
	return hist_angle;
}

// Constructor
static int create(bool multiscale, bool fixed_window)
{
	int iRet = 0;
    // Parameters equal in all cases
    lambda = 0.0001f;
    padding = 2.5f; 
    output_sigma_factor = 0.125f;
    //padding = 3.0; 
    //output_sigma_factor = 0.1042;

    // VOT
    interp_factor = 0.012f;
    sigma = 0.6f; 
    // TPAMI
    //interp_factor = 0.02;
    //sigma = 0.5; 
    cell_size = 4;

    if (multiscale) { // multiscale
        template_size = 96;
        scale_step = 1.05f;
        scale_weight = 0.95f;
        if (!fixed_window) {
            //printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
    }
    else {
        template_size = 1;
        scale_step = 1;
    }

	return iRet;
}

static void destroy()
{
	unInit();
}

__INLINE__ IMG_MAT *getHogFeatures(IMG_MAT *image, IMG_MAT** out)
{
	IMG_MAT **hog = &_FeaturesMap;
    size_patch[0] = image->height/cell_size - 2;
    size_patch[1] = image->width/cell_size - 2;
    size_patch[2] = NUM_SECTOR * 3 + 4;

	if(out != NULL)
		hog = out;

	if((*hog) == NULL)
		(*hog) = matAlloc(MAT_float, size_patch[2], size_patch[1]*size_patch[0], 1);
	else{
		UTILS_assert((*hog)->size == size_patch[2]*size_patch[1]*size_patch[0]*sizeof(float));
		(*hog)->width = size_patch[2];
		(*hog)->height = size_patch[1]*size_patch[0];
		(*hog)->step[0]= size_patch[2];
	}
	UTILS_assert((*hog) != NULL);
	memset((*hog)->data, 0, (*hog)->size);	

	GetHogFeat(image, (*hog), cell_size, NUM_SECTOR);			

	transpose((*hog));

	return (*hog);
}

// Initialize Hanning window. Function called only in the first frame.
static void createHanningMats()
{   
	int i, j;
    IMG_MAT* hann1t = matAlloc(MAT_float,size_patch[1], 1, 1);
    IMG_MAT* hann2t = matAlloc(MAT_float,1,size_patch[0], 1);
	IMG_MAT* hann3t = matAlloc(MAT_float,size_patch[1],size_patch[0], 1);

	for(i=0; i<hann1t->width; i++)
		hann1t->data[i] = 0.5f * (1 - cos(2.0f * 3.14159265358979323846f * i / (hann1t->width - 1)));
	for(i=0; i<hann2t->height; i++)
		hann2t->data[i] = 0.5f * (1 - cos(2.0f * 3.14159265358979323846f * i / (hann2t->height- 1)));

	for(j=0; j<hann3t->height; j++)
	{
		for(i=0; i<hann3t->width; i++)
			hann3t->data[j*hann3t->step[0]+i] = hann2t->data[j]*hann1t->data[i];
	}

	_hann = matAlloc(MAT_float, size_patch[0]*size_patch[1], size_patch[2], 1);

    for (i = 0; i < size_patch[2]; i++) {
        for (j = 0; j<size_patch[0]*size_patch[1]; j++) {
            _hann->data[_hann->step[0]*i+j] = hann3t->data[j];
        }
    }

	matFree(hann1t);
	matFree(hann2t);
	matFree(hann3t);
}

static void plat_compensation()
{
	int dx = abs(_offsetX);
	int dy = abs(_offsetY);
	float stepx, stepy;
	float K;
	float	dxy = sqrt(dx*dx+dy*dy);
	if(dxy< min_bias_pix/2){
		stepx	= 0.0;
		stepy = 0.0;
	}else if( dxy<= min_bias_pix){
		K = (float)dxy/min_bias_pix;
		stepx = K*(dx/dxy)*_blendValue/4;
		stepy = K*(dy/dxy)*_blendValue/4;
	}else{
		K = (float)(dxy-min_bias_pix)/(max_bias_pix-min_bias_pix);
		if(K>1.5f) K = 1.5f;
		stepx = K*(dx/dxy)*_blendValue +_blendValue/4 ;
		stepy = K*(dy/dxy)*_blendValue + _blendValue/4;
	}
	_roi.x = (_offsetX > 0) ? (_roi.x - stepx) : (_roi.x + stepx);
	_roi.y = (_offsetY > 0) ? (_roi.y + stepy) : (_roi.y - stepy);
}

// Obtain sub-window from image, with replication-padding and extract features
static IMG_MAT* getFeatures(IMG_MAT *image, bool inithann,
				float scale_adjust, UTC_RECT_float roi, IMG_MAT** out)
{
	IMG_MAT* retMat = NULL;
	IMG_MAT* subimage = NULL;
    Recti extracted_roi;
	Recti subRect;
	Sizei subSize;

    float cx = roi.x + roi.width / 2;
    float cy = roi.y + roi.height / 2;

	//_beginTm = Utils_getCurTimeInMsec();

    if (inithann) 
	{
        int padded_w = (int)(roi.width * padding +0.5f);
        int padded_h = (int)(roi.height * padding+0.5f);
        
        if (template_size > 1) {  // Fit largest dimension to the given template size
            if (padded_w >= padded_h)  //fit to width
                _scale = padded_w / (float) template_size;
            else
                _scale = padded_h / (float) template_size;

            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else {  //No template size given, use ROI size
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
        }
		_scaleBK = _scale;
        // Round to cell size and also make it even
        _tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
        _tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
    }

    extracted_roi.width = (int)(scale_adjust * _scale * _tmpl_sz.width);
    extracted_roi.height = (int)(scale_adjust * _scale * _tmpl_sz.height);

    // center roi with new size
    extracted_roi.x = (int)(cx - extracted_roi.width / 2);
    extracted_roi.y = (int)(cy - extracted_roi.height / 2);

	//Vps_printf("_roi         : %.1f, %.1f, %.1f, %.1f\n",
	//			_roi.x, _roi.y, _roi.width, _roi.height);
	//Vps_printf("scale_adjust %.1f _scale %.1f _tmpl_sz.width %d _tmpl_sz.height %d\n",
	//		scale_adjust, _scale, _tmpl_sz.width, _tmpl_sz.height);
	//Vps_printf("cx %0.1f, cy %0.1f\n", cx, cy);
	//Vps_printf("extracted_roi: %d, %d, %d, %d\n",
	//			extracted_roi.x, extracted_roi.y, extracted_roi.width, extracted_roi.height);

	//subRect.x = ((extracted_roi.x+1)&(~1));
	//subRect.y = ((extracted_roi.y+1)&(~1));
	//subRect.width = ((extracted_roi.width+1)&(~1));
	//subRect.height = ((extracted_roi.height+1)&(~1));
	subRect.x = extracted_roi.x;
	subRect.y = extracted_roi.y;
	subRect.width = extracted_roi.width;
	subRect.height = extracted_roi.height;
	subSize.width = _tmpl_sz.width;
	subSize.height = _tmpl_sz.height;
	
	subimage = matAlloc(image->dtype,
		subSize.width, subSize.height, image->channels);

	SubWindowMat(*image, subimage, subRect, subSize);
	//Vps_printf("FEA-: %d ms\n", Utils_getCurTimeInMsec()-_beginTm);

	//if((int)(scale_adjust*1000.f) != 1000)
	//	matCopy2(image, subimage);
    // HOG features
	retMat = getHogFeatures(subimage, out);
	
	matFree(subimage);
    
    if (inithann) {
        createHanningMats();
    }
	//Vps_printf("FEA+: %d ms\n", Utils_getCurTimeInMsec()-_beginTm);

    retMat = matMul(retMat, _hann);

	//_curTm = Utils_getCurTimeInMsec();
	//Vps_printf("FEA: %d ms\n", _curTm-_beginTm);

	//matDump(1, _FeaturesMap, _FeaturesMap->width*_FeaturesMap->height/4);
	
    return retMat;
}

static void convert_UC2UC_ROI(IMG_MAT	*src,	IMG_MAT	*dst,	VALID_ROI	*src_roi, VALID_ROI	*dst_roi)
{
	int i, j;
	unsigned char *psrc,	*pdst;
	assert(src != NULL && dst != NULL && src_roi != NULL && dst_roi != NULL);

#pragma UNROLL(4)
	for(j=0; j<src_roi->valid_h; j++)
	{
		psrc = (unsigned char *)(src->data_u8+(src_roi->start_y+j)*src->step[0] + src_roi->start_x);
		pdst = (unsigned char*)(dst->data_u8+(dst_roi->start_y+j)*dst->step[0] + dst_roi->start_x);
		memcpy(pdst, psrc, src_roi->valid_w*sizeof(unsigned char));
	}
}

static UInt32		frameCount = 0;
static UInt32		totalTS = 0;
static void enhROI(IMG_MAT *image, UTC_RECT_float roi, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit)
{
	int startX, startY,uiWidth, uiHeight, searchW, searchH;
	int cx, cy, rtn;
	IMG_MAT *fTmp = NULL;
	VALID_ROI	src_roi, dst_roi;

	if(image->width <= 768){
		uiWidth = (image->width< 400)?image->width:400;
		uiHeight =  (image->height< 320)?image->height:320;
	}else if(image->width <= 1280){
		uiWidth = (image->width< 600)?image->width:600;
		uiHeight =  (image->height< 512)?image->height:512;
	}else{
		uiWidth = (image->width< 640)?image->width:640;
		uiHeight =  (image->height< 512)?image->height:512;
	}

	searchW = (int)(roi.width*7);
	searchH = (int)(roi.height*7);
	searchW = ((searchW+15) &(~15));
	searchH = ((searchH+15) &(~15));

	uiWidth = (uiWidth < searchW)?uiWidth:searchW;
	uiHeight = (uiWidth < searchH)?uiHeight:searchH;

	cx = (int)(roi.x + roi.width/2);
	cy = (int)(roi.y + roi.height/2);
	startX = cx - uiWidth/2-2;
	startY = cy- uiHeight/2-2;
	if (startX < 0) startX = 0;
	if (startY < 0) startY = 0;
	startX = (startX &(~3));
	startY = (startY &(~3));

	if (startX + uiWidth > image->width-1)
		startX = image->width - uiWidth;
	if (startY + uiHeight > image->height -1)
		startY = image->height - uiHeight;

//	dbg_tmStat = dbg_getCurTimeInUsec();

#if 1
//	rtn = CLAHE_enh_omp((kz_pixel_t*)image->data_u8, image->width, image->height, startX, startY, uiWidth, uiHeight, 0, 255, uiNrX, uiNrY, 256, fCliplimit);
	rtn = CLAHE_enh_ompU8((kz_pixel_t*)image->data_u8, image->width, image->height, startX, startY, uiWidth, uiHeight, 0, 255, uiNrX, uiNrY, 256, fCliplimit);
#else
	fTmp	=	matAlloc(image->dtype, uiWidth, uiHeight, image->channels);
	src_roi.x = startX;				src_roi.y = startY;
	src_roi.width = uiWidth;	src_roi.height = uiHeight;
	dst_roi.x = 0;						dst_roi.y = 0;
	dst_roi.width = uiWidth;	dst_roi.height = uiHeight;
	convert_UC2UC_ROI(image, fTmp, &src_roi, &dst_roi);
	rtn = CLAHE_enh_omp((kz_pixel_t*)fTmp->data_u8, fTmp->width, fTmp->height, 0, 0, uiWidth, uiHeight, 0, 255, uiNrX, uiNrY, 256, fCliplimit);
	src_roi.x = 0;						src_roi.y = 0;
	src_roi.width = uiWidth;	src_roi.height = uiHeight;
	dst_roi.x = startX;				dst_roi.y = startY;
	dst_roi.width = uiWidth;	dst_roi.height = uiHeight;
	convert_UC2UC_ROI(fTmp, image, &src_roi, &dst_roi);
#endif
/*
	frameCount++;
	totalTS += (dbg_getCurTimeInUsec() - dbg_tmStat);
	if(frameCount == 100){
		printf("[DFT ENH] %.3f ms\n", totalTS/1000.f/100.f);
		frameCount = 0;
		totalTS = 0;
	}
*/
	if(rtn != 0){
		matFree(fTmp);
		printf("CLAHE_enh_omp is Error ! \n");
		assert(rtn == 0);
	}
	matFree(fTmp);
}

#if 0
static void blurFilterROI(IMG_MAT *image, UTC_RECT_float roi)
{
	IMG_MAT *fTmp = NULL;
	UTC_Rect roiRect;
	UTC_SIZE kelSize;
	int areax, areay, areaw, areah;
	areax = (int)roi.x;
	areay = (int)roi.y;
	areaw = (int)roi.width;
	areah = (int)roi.height;
	//get center pos
	areax = areax+areaw/2;
	areay = areay+areah/2;
	//set new area
	areaw = areaw*5;
	areah = areah*5;
	areaw &=(~3);
	areah &=(~3);
	if(areaw>600) areaw = 600;
	if(areah>480) areah = 480;

	areax = areax - areaw/2;
	if(areax<0)
		areax = 0;
	else if(areax > (image->width-areaw))
		areax = (image->width-areaw);
	areax &=(~3);

	areay = areay - areah/2;
	if(areay<0)
		areay = 0;
	else if(areay > (image->height-areah))
		areay = (image->height-areah);
	areay &=(~3);

	roiRect.x = areax;	roiRect.y = areay;
	roiRect.width = areaw;	roiRect.height = areah;
	kelSize.width = kelSize.height = 3;
	fTmp	=	matAlloc(image->dtype, image->width, image->height, image->channels);
	memcpy(fTmp->data_u8, image->data_u8, fTmp->size);
	BlurCR(fTmp, image, roiRect,kelSize);

	matFree(fTmp);
}
#else
static void blurFilterROI(IMG_MAT *image, UTC_RECT_float roi)
{
	cv::Mat srcMat, dstMat;
	cv::Rect roiRect;

	int areax, areay, areaw, areah;
	areax = (int)roi.x;
	areay = (int)roi.y;
	areaw = (int)roi.width;
	areah = (int)roi.height;
	//get center pos
	areax = areax+areaw/2;
	areay = areay+areah/2;
	//set new area
	areaw = (int)(areaw*7);
	areah = (int)(areah*7);
	areaw &=(~3);
	areah &=(~3);
	if(areaw>600) areaw = 600;
	if(areah>512) areah = 512;

	areax = areax - areaw/2;
	if(areax<0)
		areax = 0;
	else if(areax > (image->width-areaw))
		areax = (image->width-areaw);
	areax &=(~3);

	areay = areay - areah/2;
	if(areay<0)
		areay = 0;
	else if(areay > (image->height-areah))
		areay = (image->height-areah);
	areay &=(~3);

	roiRect.x = areax;	roiRect.y = areay;
	roiRect.width = areaw;	roiRect.height = areah;
	srcMat = cv::Mat(image->height, image->width, CV_8UC1);
	dstMat = cv::Mat(image->height, image->width, CV_8UC1, image->data_u8);
	cv::Mat dysrc, dydst;
	dysrc = srcMat(roiRect);
	dydst = dstMat(roiRect);
	dydst.copyTo(dysrc);
	blur(dysrc, dydst, cv::Size(3,3));
}
#endif

static char *strMode[]={
		"SEARCH_MODE_NO",
		"SEARCH_MODE_ALL",
		"SEARCH_MODE_LEFT",
		"SEARCH_MODE_RIGHT",
		"SEARCH_MODE_NEAREST",
};
// Initialize tracker 
static int init(UTC_RECT_float roi, IMG_MAT *image)
{
	int iRet = 0;
	int i;
	IMG_MAT *fTmp = NULL;

	if(first_print == 0){
		printf("[DFT] ver: 00.00.04.2 -- 21/5/18\n");

		if(gSerchMode >= SEARCH_MODE_ALL && gSerchMode <SEARCH_MODE_MAX){
			printf("Track SearchMode:%s\n",strMode[gSerchMode]);
		}else{
			printf("Do not Set SearchMode!\n");
			gSerchMode = SEARCH_MODE_ALL;
		}
		first_print = 1;
	}

	UTILS_assert(roi.width >= 0 && roi.height >= 0);

    _roi  = roi;

    _tskMap = matAlloc(image->dtype, image->width, image->height, image->channels);

    fTmp = getFeatures(image, 1, 1.0f, _roi, NULL);
	
	_tmpl = matAlloc(fTmp->dtype, fTmp->width, fTmp->height, fTmp->channels);
	matCopy(_tmpl, fTmp);

	_origtmpl= matAlloc(fTmp->dtype, fTmp->width, fTmp->height, fTmp->channels);//20180421
	matCopy(_origtmpl, fTmp);

	_prob = matAlloc(MAT_float, size_patch[1], size_patch[0], 2);
	createGaussianPeak(*_prob, size_patch[0], size_patch[1], padding, output_sigma_factor);
	
    _alphaf = matAlloc(MAT_float, size_patch[1], size_patch[0], 2);

    _origalphaf = matAlloc(MAT_float, size_patch[1], size_patch[0], 2);//20180421

    _peakMat = matAlloc(MAT_float, size_patch[1], size_patch[0], 1);//20180517

	for(i=0; i<SAVE_NUM; i++)
	{
		_feature_tmpl[i] = matAlloc(MAT_float, _tmpl->width, _tmpl->height, _tmpl->channels);
		_feature_alphaf[i] = matAlloc(MAT_float, _alphaf->width, _alphaf->height, _alphaf->channels);
		_feature_peak[i] = 0.0f;
	}
	featIndx = 0;
	lostFrame = 0;
	utcStatus = 1;

	// train with initial frame
    trainCR(*fTmp, *_prob, _tmpl, _alphaf, 1.0, lambda, size_patch, true, sigma);
    matCopy(_origalphaf, _alphaf);//20180421

	_feature_peak[featIndx ] = 1.0f;
	matCopy(_feature_tmpl[featIndx], _tmpl);
	matCopy(_feature_alphaf[featIndx], _alphaf);
	featIndx++;
	if(featIndx == SAVE_NUM)
		featIndx = 0;	

	return iRet;
}

static void unInit()
{
	int i;
	
	matFree(_hann);
	matFree(_tmpl);
	matFree(_alphaf);
	matFree(_prob);
	matFree(_FeaturesMap);
	matFree(_tskMap);
	matFree(_origtmpl);
	matFree(_origalphaf);
	matFree(_peakMat);
	_hann = NULL;
	_tmpl = NULL;
	_alphaf = NULL;
	_prob = NULL;
	_FeaturesMap = NULL;
	_tskMap = NULL;
	_origtmpl = NULL;
	_origalphaf = NULL;
	_peakMat = NULL;

	for(i=0; i<SAVE_NUM; i++)
	{
		matFree(_feature_tmpl[i]);
		_feature_tmpl[i] = NULL;
		matFree(_feature_alphaf[i]);
		_feature_alphaf[i] = NULL;
		_feature_peak[i] = 0.0f;
	}
}

// Detect object in the current frame.
static PointfCR detect(IMG_MAT templ, IMG_MAT feature, float *pPeak_value, IMG_MAT *resMat)
{
	PointfCR pt;
	IMG_MAT *res;
	float  peakValue;

	//_beginTm = Utils_getCurTimeInMsec();

	res = matAlloc(MAT_float, size_patch[1], size_patch[0], 1);
	detectCR(templ, feature, res, *_alphaf, size_patch, true, sigma);

	getMinMaxValue(*res, &pt, &peakValue);
	if(pPeak_value != NULL)
		*pPeak_value = peakValue;

	if(resMat != NULL){
		matCopy(resMat, res);
	}

	matFree(res);

	//_curTm = Utils_getCurTimeInMsec();
	//Vps_printf("DET: %d ms\n", _curTm-_beginTm);	
	
    return pt;
}

static void calAveOffset(UTC_RECT_float	*aveRect)
{
	int i;
	aveRect->x= aveRect->y = 0.f;
	if(_platType == tPLT_WRK){ //close loop
		for(i=0; i<trkPosNmu; i++){
			aveRect->x += fabs( _track_pos[i].x - pTrkObj->axisX);
			aveRect->y += fabs( _track_pos[i].y - pTrkObj->axisY);
		}
	}else{//open loop
		aveRect->x = fabs(_track_pos[trkPosNmu-1].x -  _track_pos[0].x);
		aveRect->y = fabs(_track_pos[trkPosNmu-1].y -  _track_pos[0].y);
	}
	aveRect->x /=trkPosNmu;		aveRect->y /=trkPosNmu;
}

// Update position based on the new frame
static UTC_RECT_float update(IMG_MAT *image)
{
	float cx, cy;
	float peak_value;
	PointfCR res;
	IMG_MAT *features;
	UTC_RECT_float tmpRect = _roi;
	IMG_MAT	*resMat =NULL;
	
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image->width - 1) _roi.x = image->width - 2;
	if (_roi.y >= image->height - 1) _roi.y = image->height - 2;

	cx = _roi.x + _roi.width / 2.0f;
	cy = _roi.y + _roi.height / 2.0f;

	_peakRoi = _roi;

	//matDump(0, _tmpl, _tmpl->width*_tmpl->height/4);

	features = getFeatures(image, 0, 1.0f, _roi, NULL);

	resMat = matAlloc(MAT_float, size_patch[1], size_patch[0], 1);

	res = detect(*_tmpl, *features, &peak_value, /*NULL*/resMat);

	matCopy(_peakMat, resMat);

    if (scale_step != 1) {
        // Test at a smaller _scale
        float new_peak_value;
        PointfCR new_res;

        // Test at a small _scale
		if(_roi.width > _roiMinWidth && _roi.height > _roiMinWidth){
			features = getFeatures(image, 0, 1.0f/ scale_step, _roi, NULL);
			new_res = detect(*_tmpl, *features, &new_peak_value, /*NULL*/resMat);

			if (scale_weight * new_peak_value > peak_value && scale_weight * new_peak_value > gDynamicParam.occlusion_thred) {
				res = new_res;
				peak_value = new_peak_value;
				_scale /= scale_step;
				_roi.width /= scale_step;
				_roi.height /= scale_step;
				matCopy(_peakMat, resMat);
			}
		}

        // Test at a bigger _scale
        if(_roi.width < _roiMaxWidth){
			features = getFeatures(image, 0, scale_step, _roi, NULL);
			new_res = detect(*_tmpl, *features, &new_peak_value, /*NULL*/resMat);

			if (scale_weight * new_peak_value > peak_value && scale_weight * new_peak_value > gDynamicParam.occlusion_thred) {
				res = new_res;
				peak_value = new_peak_value;
				_scale *= scale_step;
				_roi.width *= scale_step;
				_roi.height *= scale_step;
				matCopy(_peakMat, resMat);
			}
        }
    }

	opt_peak_value = peak_value;

	if(opt_peak_value < gDynamicParam.occlusion_thred*0.68){
		res.x = res.y = 0.0;
	}

	_peakRes = res;

	// Adjust by cell size and _scale
#if 1
	_roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

	if (_roi.x >= image->width - 1) _roi.x = image->width - 1;
	if (_roi.y >= image->height - 1) _roi.y = image->height - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

	UTILS_assert(_roi.width >= 0 && _roi.height >= 0);

	features = getFeatures(image, 0, 1.0f, _roi, NULL);

	trainCR(*features, *_prob, _tmpl, _alphaf,interp_factor, lambda, size_patch, true, sigma);

#endif

	_scaleBK = _scale;
	_roiBK = _roi;

	if(opt_peak_value < gDynamicParam.occlusion_thred)
	{
		lostFrame++;
		utcStatus = 0;
		if(trkPosNmu>=TRK_POS_NUM && pTrkObj != NULL && _bAveTrkPos){
			UTC_RECT_float	aveRect;
			calAveOffset(&aveRect);
			if(_platType == tPLT_WRK){ //close loop
				if(fabs(_roi.x+_roi.width/2-pTrkObj->axisX)>(5*aveRect.x) || fabs(_roi.y+_roi.height/2-pTrkObj->axisY)>(5*aveRect.y)){
					_roi = tmpRect;
					_roiBK = _roi;
//					printf("%s: Track pos is abnormal!	\n",__func__);
				}
			}else{//open loop
				if(fabs(_roi.x-tmpRect.x)>(5*aveRect.x) || fabs(_roi.y-tmpRect.y)>(5*aveRect.y)){
					_roi = tmpRect;
					_roiBK = _roi;
				}
			}
		}
	}
	else
	{
		_feature_peak[featIndx ] = opt_peak_value;
		matCopy(_feature_tmpl[featIndx], _tmpl);
		matCopy(_feature_alphaf[featIndx], _alphaf);
		featIndx++;
		if(featIndx == SAVE_NUM)
			featIndx = 0;

		lostFrame = 0;
		utcStatus  = 1;

		if(trkPosNmu<TRK_POS_NUM){
			_track_pos[trkPosNmu].x = _roi.x + _roi.width/2;
			_track_pos[trkPosNmu].y = _roi.y + _roi.height/2;
			trkPosNmu++;
		}else{
			memmove(_track_pos, (_track_pos+1), sizeof(PointfCR)*(TRK_POS_NUM-1));
			_track_pos[TRK_POS_NUM-1].x = _roi.x + _roi.width/2;
			_track_pos[TRK_POS_NUM-1].y = _roi.y + _roi.height/2;
		}
	}
	if(resMat != NULL){
		matFree(resMat);
		resMat = NULL;
	}
    return _roi;
}

static float reSearchOpenMP(IMG_MAT *image, UTC_RECT_float rectBak)
{
	int i,j,k;
	float peak_value[25],max_value = 0.0;
	PointfCR opt_res, res[25];
	UTC_RECT_float rect[25];
	int iOpt = 0;
	float cx, cy;

	opt_peak_value_search = 0.0;
	opt_roi_search = rectBak;

	memset(peak_value, 0, sizeof(peak_value));

	opt_res.x = opt_res.y = 0.0;
	cx = rectBak.x + rectBak.width/2;
	cy = rectBak.y + rectBak.height/2;

	for(j=0; j<3; j++)
	{
		rect[j].x = rectBak.x - 2*rectBak.width;
		rect[j].y = rectBak.y + (j-1)*rectBak.height;
		rect[j+3].x = rectBak.x + 2*rectBak.width;
		rect[j+3].y = rectBak.y + (j-1)*rectBak.height;
	}
	{
		rect[6].x = rectBak.x - rectBak.width;
		rect[6].y = rectBak.y - 2*rectBak.height;
		rect[7].x = rectBak.x - rectBak.width;
		rect[7].y = rectBak.y + 2*rectBak.height;
		rect[8].x = rectBak.x + rectBak.width;
		rect[8].y = rectBak.y - 2*rectBak.height;
		rect[9].x = rectBak.x + rectBak.width;
		rect[9].y = rectBak.y + 2*rectBak.height;
	}

#pragma omp parallel for
	for(k=0; k<10; k++)
	{
		bool  bRun = true;
		rect[k].width = rectBak.width;
		rect[k].height = rectBak.height;

		if (rect[k].x + rect[k].width <= 0 || rect[k].y + rect[k].height <= 0 ) {
			peak_value[k] = 0.0;
			bRun = false;
		}
		if (rect[k].x >= image->width - 1 || rect[k].y >= image->height - 1) {
			peak_value[k] = 0.0;
			bRun = false;
		}

		if (rect[k].x < 0) rect[k].x = 0;
		if (rect[k].y < 0) rect[k].y = 0;
		if (rect[k].x + rect[k].width > image->width-1)
			rect[k].x = image->width -1 - rect[k].width;
		if (rect[k].y + rect[k].height > image->height -1)
			rect[k].y = image->height -1 - rect[k].height;

		if(bRun){
			IMG_MAT *features = NULL;
			features = getFeatures(image, 0, 1.0f, rect[k], &features);
			res[k] = detect(*_tmpl, *features, &peak_value[k], NULL);
			matFree(features);
		}
	}

	for(k=0; k<10; k++)
	{
		if(peak_value[k] > max_value){
			max_value = peak_value[k];
			opt_res = res[k];
			cx = rect[k].x + rect[k].width / 2.0f;
			cy = rect[k].y + rect[k].height / 2.0f;
			iOpt = k;
		}
	}

	// Adjust by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float) opt_res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float) opt_res.y * cell_size * _scale);

	if (_roi.x < 0) _roi.x = 0;
	if (_roi.y < 0) _roi.y = 0;
	if (_roi.x + _roi.width > image->width-1)
		_roi.x = image->width -1 - _roi.width;
	if (_roi.y + _roi.height > image->height -1)
		_roi.y = image->height -1 - _roi.height;

	assert(_roi.width >= 0 && _roi.height >= 0);
	if(max_value > adaptive_acq_thred){
		utcStatus = 1;
		lostFrame = 0;
		opt_peak_value = max_value;
		_roiBK = _roi;
	}
	return max_value;
}

static UTC_RECT_float searchOpenMP(IMG_MAT *image)
{
	UTC_RECT_float recTmpBK;
	int i,j,k;
	float peak_value[25],max_value = 0.0, big_value = 0.f;
	PointfCR opt_res, res[25];
	UTC_RECT_float rect[25];
	UTC_RECT_float	aveRect;
	int iOpt = 0;
	float cx, cy;
	int lostFrameBK = lostFrame;

#if 1
	if(_platType == tPLT_WRK){ //close loop
		_roi.x = (pTrkObj->axisX-_roi.width/2);
		_roi.y = (pTrkObj->axisY-_roi.height/2);
	}
#endif

	opt_peak_value_search = 0.0;
	opt_roi_search = _roi;

	memset(peak_value, 0, sizeof(peak_value));
	
	recTmpBK = _roi;
	opt_res.x = opt_res.y = 0.0;
	cx = _roi.x + _roi.width/2;
	cy = _roi.y + _roi.height/2;

	for(j=0; j<3; j++)
	{
		for(i=0; i<3; i++)
		{
			rect[j*3+i].x = recTmpBK.x + (i-1)*recTmpBK.width;
			rect[j*3+i].y = recTmpBK.y + (j-1)*recTmpBK.height;
		}
	}

#pragma omp parallel for
	for(k=0; k<9; k++)
	{
		bool  bRun = true;
		rect[k].width = recTmpBK.width;
		rect[k].height = recTmpBK.height;

		if (rect[k].x + rect[k].width <= 0 || rect[k].y + rect[k].height <= 0 ) {
			peak_value[k] = 0.0;
			bRun = false;
		}
		if (rect[k].x >= image->width - 1 || rect[k].y >= image->height - 1) {
			peak_value[k] = 0.0;
			bRun = false;
		}

		if (rect[k].x < 0) rect[k].x = 0;
		if (rect[k].y < 0) rect[k].y = 0;
		if (rect[k].x + rect[k].width > image->width-1)
			rect[k].x = image->width -1 - rect[k].width;
		if (rect[k].y + rect[k].height > image->height -1)
			rect[k].y = image->height -1 - rect[k].height;

		if(bRun){
			IMG_MAT *features = NULL;
			features = getFeatures(image, 0, 1.0f, rect[k], &features);
			res[k] = detect(*_tmpl, *features, &peak_value[k], NULL);
			matFree(features);
		}
	}

	for(k=0; k<9; k++)
	{
		if(peak_value[k] > max_value){
			max_value = peak_value[k];
			opt_res = res[k];
			cx = rect[k].x + rect[k].width / 2.0f;
			cy = rect[k].y + rect[k].height / 2.0f;
			iOpt = k;
		}
	}

	// Adjust by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float) opt_res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float) opt_res.y * cell_size * _scale);

	if (_roi.x < 0) _roi.x = 0;
	if (_roi.y < 0) _roi.y = 0;
	if (_roi.x + _roi.width > image->width-1)
		_roi.x = image->width -1 - _roi.width;
	if (_roi.y + _roi.height > image->height -1)
		_roi.y = image->height -1 - _roi.height;

	assert(_roi.width >= 0 && _roi.height >= 0);

	TRAJECTORY_PARAM	*pTrajParam = &gSceneMVHdl->m_trajParam;
	PointfCR histDelta = gSceneMVHdl->m_trajParam.m_histDelta;
	histDelta.x = -histDelta.x;//direction opposite
	float histSqrt = sqrt(histDelta.x*histDelta.x + histDelta.y*histDelta.y);
	float		mvThred;
	if(image->width <= 768){
		mvThred = _histMvThred/2;
	}else if(image->width <= 1280){
		mvThred = _histMvThred;
	}else{
		mvThred = _histMvThred*1.5;
	}

	if(max_value > adaptive_acq_thred){
		utcStatus = 1;
		opt_peak_value = max_value;

		if(trkPosNmu>=TRK_POS_NUM && pTrkObj != NULL && lostFrame<(_LostFrmThred*2/3) && _bAveTrkPos){
			calAveOffset(&aveRect);
			if(_platType == tPLT_WRK){ //close loop
				if(fabs(_roi.x+_roi.width/2-pTrkObj->axisX)>((lostFrame/4+5)*aveRect.x) || fabs(_roi.y+_roi.height/2-pTrkObj->axisY)>((lostFrame/4+5)*aveRect.y)){
					_roi = recTmpBK;
					_roiBK = _roi;
					utcStatus = 0;
				}
			}else{//open loop
				if(fabs(_roi.x-recTmpBK.x)>((lostFrame/4+5)*aveRect.x) || fabs(_roi.y-recTmpBK.y)>((lostFrame/4+5)*aveRect.y)){
					_roi = recTmpBK;
					_roiBK = _roi;
					utcStatus = 0;
				}
			}
		}
		if(utcStatus == 1){
			lostFrame = 0;
			_roiBK = _roi;
		}

	}else{
		if(_bCalSceneMV && _bBigSearch && lostFrame>_LostFrmThred ){
			if( histSqrt>(mvThred*4))
				big_value = reSearchOpenMP(image, recTmpBK);
		}else if(_bBigSearch && lostFrame>_LostFrmThred ){
			big_value = reSearchOpenMP(image, recTmpBK);
		}
	}
	opt_peak_value_search = (big_value>max_value)?big_value:max_value;
	opt_roi_search =_roi;

	if(utcStatus == 1 && _bCalSceneMV && pTrajParam->m_histNum>15 && pTrkObj != NULL){
		PointfCR 	trkPt;
		float offAxis,diag;

		trkPt.x = _roi.x + _roi.width/2;
		trkPt.y = _roi.y + _roi.height/2;
		offAxis = sqrt((trkPt.x-pTrkObj->axisX)*(trkPt.x-pTrkObj->axisX)+(trkPt.y-pTrkObj->axisY)*(trkPt.y-pTrkObj->axisY));
		diag = sqrt(_roi.width*_roi.width+_roi.height*_roi.height);

		trkPt.x = trkPt.x - pTrkObj->axisX;
		trkPt.y = pTrkObj->axisY - trkPt.y;

		if(histSqrt > mvThred){
			float histAngle = _getAngle(histDelta);
			float trkAngle = _getAngle(trkPt);

			if(histAngle>=0.0 && histAngle<=90.0 && trkAngle>=270.0 && trkAngle<=360.0){
				histAngle+= 360.0;
			}else if(trkAngle>=0.0 && trkAngle<=90.0 && histAngle>=270.0 && histAngle<=360.0){
				trkAngle+= 360.0;
			}

			if(fabs(histAngle-trkAngle)<_trkAngleThred || (fabs(histAngle-trkAngle) >(180-_trkAngleThred/2) && fabs(histAngle-trkAngle) <(180+_trkAngleThred/2) ) ||
				offAxis <(diag*0.75)){
				;//normal status
			}else if(_bHighOpt && opt_peak_value_search>0.64 && offAxis<(diag*1.75)){
				;//normal status
			}else{
				utcStatus = 0;
				lostFrame = lostFrameBK;
				_roi = recTmpBK;
				_roiBK = _roi;
			}
		}
	}

	if(utcStatus == 0)
	{
		_roi = recTmpBK;
		lostFrame++;
	}

	return _roi;
}

static UTC_RECT_float searchOpenMP_LR(IMG_MAT *image, int serchMode)
{
	UTC_RECT_float recTmpBK;
	int k;
	float peak_value[25],max_value = 0.0;
	PointfCR opt_res, res[25];
	UTC_RECT_float rect[25];
	int iOpt = 0;
	float cx, cy;
	bool  bRun = true;

	opt_peak_value_search = 0.0;
	opt_roi_search = _roi;

	memset(peak_value, 0, sizeof(peak_value));
	//memset(mats, 0, sizeof(mats));

	recTmpBK = _roi;
	opt_res.x = opt_res.y = 0.0;
	cx = _roi.x + _roi.width/2;
	cy = _roi.y + _roi.height/2;

	if(serchMode == SEARCH_MODE_LEFT){
		rect[0].x =  recTmpBK.x - recTmpBK.width;
		rect[1].x =  recTmpBK.x - recTmpBK.width;
		rect[2].x =  recTmpBK.x - recTmpBK.width;
		rect[0].y =  recTmpBK.y - recTmpBK.height;
		rect[1].y =  recTmpBK.y ;
		rect[2].y =  recTmpBK.y + recTmpBK.height;

	}else if(serchMode == SEARCH_MODE_RIGHT){
		rect[0].x =  recTmpBK.x + recTmpBK.width;
		rect[1].x =  recTmpBK.x + recTmpBK.width;
		rect[2].x =  recTmpBK.x + recTmpBK.width;
		rect[0].y =  recTmpBK.y - recTmpBK.height;
		rect[1].y =  recTmpBK.y ;
		rect[2].y =  recTmpBK.y + recTmpBK.height;
	}
	rect[3].x =  recTmpBK.x;
	rect[4].x =  recTmpBK.x ;
	rect[5].x =  recTmpBK.x ;
	rect[3].y =  recTmpBK.y - recTmpBK.height;
	rect[4].y =  recTmpBK.y ;
	rect[5].y =  recTmpBK.y + recTmpBK.height;

	#pragma omp parallel for
	for(k=0; k<6; k++)
	{
		rect[k].width = recTmpBK.width; 
		rect[k].height = recTmpBK.height;

		if (rect[k].x + rect[k].width <= 0 || rect[k].y + rect[k].height <= 0 ) {
			peak_value[k] = 0.0;
			bRun = false;
		}
		if (rect[k].x >= image->width - 1 || rect[k].y >= image->height - 1) {
			peak_value[k] = 0.0;
			bRun = false;
		}

		if (rect[k].x < 0) rect[k].x = 0;
		if (rect[k].y < 0) rect[k].y = 0;
		if (rect[k].x + rect[k].width > image->width-1)
			rect[k].x = image->width -1 - rect[k].width;
		if (rect[k].y + rect[k].height > image->height -1)
			rect[k].y = image->height -1 - rect[k].height;

		if(bRun){
			IMG_MAT *features = NULL;
			features = getFeatures(image, 0, 1.0f, rect[k], &features);
			res[k] = detect(*_tmpl, *features, &peak_value[k], NULL);
			matFree(features);
		}
	}

	for(k=0; k<6; k++)
	{
		if(peak_value[k] > max_value){
			max_value = peak_value[k];
			opt_res = res[k];
			cx = rect[k].x + rect[k].width / 2.0f;
			cy = rect[k].y + rect[k].height / 2.0f;
			iOpt = k;
		}
	}

	// Adjust by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float) opt_res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float) opt_res.y * cell_size * _scale);

	if (_roi.x < 0) _roi.x = 0;
	if (_roi.y < 0) _roi.y = 0;
	if (_roi.x + _roi.width > image->width-1)
		_roi.x = image->width -1 - _roi.width;
	if (_roi.y + _roi.height > image->height -1)
		_roi.y = image->height -1 - _roi.height;

	assert(_roi.width >= 0 && _roi.height >= 0);

	if(max_value > adaptive_acq_thred){
		utcStatus = 1;
		lostFrame = 0;
		recTmpBK = _roi;
		opt_peak_value = max_value;
	}else{
		;
	}
	opt_peak_value_search = max_value;
	opt_roi_search =_roi;
	if(utcStatus == 0)
	{
		_roi = recTmpBK;
		lostFrame++;
	}
	
	return _roi;
}

static UTC_RECT_float searchOpenMP_Nearest(IMG_MAT *image)
{
	UTC_RECT_float recTmpBK;
	int k;
	float peak_value[25],max_value = 0.0;
	PointfCR opt_res, res[25];
	UTC_RECT_float rect[25];
	int iOpt = 0;
	float cx, cy;
	bool  bRun = true;

	opt_peak_value_search = 0.0;
	opt_roi_search = _roi;

	memset(peak_value, 0, sizeof(peak_value));

	recTmpBK = _roi;
	opt_res.x = opt_res.y = 0.0;
	cx = _roi.x + _roi.width/2;
	cy = _roi.y + _roi.height/2;

	k = 0;
	rect[k] = recTmpBK;

	if (rect[k].x + rect[k].width <= 0 || rect[k].y + rect[k].height <= 0 ) {
		peak_value[k] = 0.0;
		bRun = false;
	}
	if (rect[k].x >= image->width - 1 || rect[k].y >= image->height - 1) {
		peak_value[k] = 0.0;
		bRun = false;
	}

	if (rect[k].x < 0) rect[k].x = 0;
	if (rect[k].y < 0) rect[k].y = 0;
	if (rect[k].x + rect[k].width > image->width-1)
		rect[k].x = image->width -1 - rect[k].width;
	if (rect[k].y + rect[k].height > image->height -1)
		rect[k].y = image->height -1 - rect[k].height;

	if(bRun){
		IMG_MAT *features = NULL;
		features = getFeatures(image, 0, 1.0f, rect[k], &features);
		res[k] = detect(*_tmpl, *features, &peak_value[k], NULL);
		matFree(features);
	}

	max_value = peak_value[k];
	opt_res = res[k];
	cx = rect[k].x + rect[k].width / 2.0f;
	cy = rect[k].y + rect[k].height / 2.0f;

	// Adjust by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float) opt_res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float) opt_res.y * cell_size * _scale);

	if (_roi.x < 0) _roi.x = 0;
	if (_roi.y < 0) _roi.y = 0;
	if (_roi.x + _roi.width > image->width-1)
		_roi.x = image->width -1 - _roi.width;
	if (_roi.y + _roi.height > image->height -1)
		_roi.y = image->height -1 - _roi.height;

	assert(_roi.width >= 0 && _roi.height >= 0);

	if(max_value > adaptive_acq_thred){
		utcStatus = 1;
		lostFrame = 0;
		recTmpBK = _roi;
		opt_peak_value = max_value;
	}else{
		;
	}
	opt_peak_value_search = max_value;
	opt_roi_search =_roi;
	if(utcStatus == 0)
	{
		_roi = recTmpBK;
		lostFrame++;
	}

	return _roi;
}

static void updataROI(void)
{
	int maxIdx = -1;
	float maxpeak = 0.0001;
	int i;
	int size =SAVE_NUM;
	for(i=0; i<size; i++)
	{
		if(_feature_peak[i] > maxpeak){
			maxpeak = _feature_peak[i];
			maxIdx = i;
		}
	}
	if(size > 0 && maxIdx !=-1){
		matCopy(_tmpl, _feature_tmpl[maxIdx]);
		matCopy(_alphaf, _feature_alphaf[maxIdx]);
		_scale = _scaleBK;
		_roi = _roiBK;
		_maxpeak = maxpeak;
	}
}

static void updataROI2(void)
{
	int maxIdx = -1;
	float maxpeak = 0.0001;
	int i;
	int size =SAVE_NUM;
	for(i=0; i<size; i++)
	{
		if(_feature_peak[i] > maxpeak){
			maxpeak = _feature_peak[i];
			maxIdx = i;
		}
	}
	if(size > 0 && maxIdx !=-1){
		_maxpeak = maxpeak;
	}

	matCopy(_tmpl, _origtmpl);
	matCopy(_alphaf, _origalphaf);
	_scale = _scaleBK;
	_roi = _roiBK;
}

static float getMaxPeak()
{
	int maxIdx = -1;
	float maxpeak = 0.0001;
	int i;
	int size =SAVE_NUM;
	for(i=0; i<size; i++)
	{
		if(_feature_peak[i] > maxpeak){
			maxpeak = _feature_peak[i];
			maxIdx = i;
		}
	}
	return maxpeak;
}

/**********************************************************************
*
*
*
*/
typedef struct _back_thr_obj{

	OSA_ThrHndl		thrHandleProc;
	OSA_SemHndl	procNotifySem;
	OSA_SemHndl	procWaitSem;
	volatile bool	exitProcThread;
	bool					initFlag;
}BACK_TrkThrObj;

static BACK_TrkThrObj	gBackThrTsk;

static void *mainProcTsk(void *);

static int BackTrk_threadCreate(void)
{
	int iRet = OSA_SOK;

	iRet = OSA_semCreate(&gBackThrTsk.procNotifySem ,1,0) ;
	OSA_assert(iRet == OSA_SOK);

	iRet = OSA_semCreate(&gBackThrTsk.procWaitSem ,1,0) ;
	OSA_assert(iRet == OSA_SOK);

	gBackThrTsk.exitProcThread = false;

	gBackThrTsk.initFlag = true;

	iRet = OSA_thrCreate(&gBackThrTsk.thrHandleProc, mainProcTsk, 0, 0, &gBackThrTsk);

	return iRet;
}
static int BackTrk_threadDestroy(void)
{
	int iRet = OSA_SOK;

	gBackThrTsk.exitProcThread = true;
	OSA_semSignal(&gBackThrTsk.procNotifySem);
	OSA_semSignal(&gBackThrTsk.procWaitSem);

	iRet = OSA_thrDelete(&gBackThrTsk.thrHandleProc);

	gBackThrTsk.initFlag = false;
	OSA_semDelete(&gBackThrTsk.procNotifySem);
	OSA_semDelete(&gBackThrTsk.procWaitSem);

	return iRet;
}

static int create(bool multiscale, bool fixed_window);
static void destroy();

UTCTRACK_HANDLE CreateUtcTrk()
{
	UTCTRACK_OBJ* pUtcTrkObj = &UtcTrk0bj;

	pTrkObj = pUtcTrkObj;

	memset(pUtcTrkObj, 0, sizeof(UTCTRACK_OBJ));

	pUtcTrkObj->axisX = 1920/2;
	pUtcTrkObj->axisY = 1080/2;

	pUtcTrkObj->m_bInited = true;

	gDynamicParam.occlusion_thred = OCLUSION_THRED;
	gDynamicParam.retry_acq_thred = RETRY_ACQ_THRED;

	create(true, false);
	//create(false, true);
	
	if(gConRegHdl == NULL){
		bool bSuccess = CreatConRegObj(&gConRegHdl);
		assert(bSuccess);
	}

	if(gSceneMVHdl == NULL){
		gSceneMVHdl = CreateSceneHdl();
	}

	if(gHistTrkHdl == NULL){
		gHistTrkHdl = CreateHistTrk(false, true);
	}

	memset(&gSubTrkParam, 0, sizeof(SUB_TRK_PARAM));

	BackTrk_threadCreate();

#ifdef BGFG_CR
	gMVDetectObj.creat();
	gMVDetectObj.setDetectShadows(true);//检测影子
	gMVDetectObj.setShadowThreshold(_detectfTau);
//	gMVDetectObj.setShadowValue(127);
//	gMVDetectObj.setHistory(DETECTOR_HISTORY_FRAMES);
#else
	gMVDetectObj.creat(500, 16, false);
#endif

	return (UTCTRACK_HANDLE)pUtcTrkObj;
}

void DestroyUtcTrk(UTCTRACK_HANDLE handle)
{
	UTCTRACK_OBJ* pUtcTrkObj = handle; 

	UTILS_assert(pUtcTrkObj == &UtcTrk0bj);

	BackTrk_threadDestroy();

	memset(pUtcTrkObj, 0, sizeof(UTCTRACK_OBJ));
	
	destroy();

	DestroyRegObj(&gConRegHdl);

	if(gSceneMVHdl != NULL){
		CloseSceneHdl(gSceneMVHdl);
		gSceneMVHdl = NULL;
	}

	if(gHistTrkHdl != NULL){
		DestroySceneTrk(gHistTrkHdl);
		gHistTrkHdl = NULL;
	}

	gMVDetectObj.destroy();

	return ;
}

static UTC_RECT_float _trkserchMP(IMG_MAT *image)
{
	UTC_RECT_float recTmpBK;
	int i,j,k;
	float peak_value[25],max_value = 0.0;
	PointfCR opt_res, res[25];
	UTC_RECT_float rect[25];
	int iOpt = 0;
	float cx, cy;

	opt_peak_value_search = 0.0;
	opt_roi_search = _roi;

	memset(peak_value, 0, sizeof(peak_value));
	
	recTmpBK = _roi;
	opt_res.x = opt_res.y = 0.0;
	cx = _roi.x + _roi.width/2;
	cy = _roi.y + _roi.height/2;

	for(j=0; j<3; j++)
	{
		for(i=0; i<3; i++)
		{
			rect[j*3+i].x = recTmpBK.x + (i-1)*recTmpBK.width;
			rect[j*3+i].y = recTmpBK.y + (j-1)*recTmpBK.height;
		}
	}

	#pragma omp parallel for
	for(k=0; k<9; k++)
	{
		bool  bRun = true;
		rect[k].width = recTmpBK.width;
		rect[k].height = recTmpBK.height;

		if (rect[k].x + rect[k].width <= 0 || rect[k].y + rect[k].height <= 0 ) {
			peak_value[k] = 0.0;
			bRun = false;
		}
		if (rect[k].x >= image->width - 1 || rect[k].y >= image->height - 1) {
			peak_value[k] = 0.0;
			bRun = false;
		}

		if (rect[k].x < 0) rect[k].x = 0;
		if (rect[k].y < 0) rect[k].y = 0;
		if (rect[k].x + rect[k].width > image->width-1)
			rect[k].x = image->width -1 - rect[k].width;
		if (rect[k].y + rect[k].height > image->height -1)
			rect[k].y = image->height -1 - rect[k].height;

		if(bRun){
			IMG_MAT *features = NULL;
			features = getFeatures(image, 0, 1.0f, rect[k], &features);
			res[k] = detect(*_tmpl, *features, &peak_value[k], NULL);
			matFree(features);
		}
	}

	for(k=0; k<9; k++)
	{
		if(peak_value[k] > max_value){
			max_value = peak_value[k];
			opt_res = res[k];
			cx = rect[k].x + rect[k].width / 2.0f;
			cy = rect[k].y + rect[k].height / 2.0f;
			iOpt = k;
		}
	}

	// Adjust by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float) opt_res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float) opt_res.y * cell_size * _scale);

	if (_roi.x < 0) _roi.x = 0;
	if (_roi.y < 0) _roi.y = 0;
	if (_roi.x + _roi.width > image->width-1)
		_roi.x = image->width -1 - _roi.width;
	if (_roi.y + _roi.height > image->height -1)
		_roi.y = image->height -1 - _roi.height;

	UTILS_assert(_roi.width >= 0 && _roi.height >= 0);

	if(max_value > /*adaptive_acq_thred*/gDynamicParam.occlusion_thred){
		utcStatus = 1;
		lostFrame = 0;
		recTmpBK = _roi;
		opt_peak_value = max_value;
	}else{
		utcStatus = 0 ;
	}
	opt_peak_value_search = max_value;
	opt_roi_search =_roi;
	if(utcStatus == 0)
	{
		_roi = recTmpBK;
		lostFrame++;
	}

	return _roi;
}

static UTC_RECT_float _trkseach(UTCTRACK_OBJ *pUtcTrkObj,IMG_MAT *image)
{
	float cx, cy;
	float peak_value = 0.f;
	PointfCR res;
	IMG_MAT *features;
	UTC_RECT_float rcResult;
	PointfCR	 centPt;
	
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image->width - 1) _roi.x = image->width - 2;
	if (_roi.y >= image->height - 1) _roi.y = image->height - 2;

	cx = _roi.x + _roi.width / 2.0f;
	cy = _roi.y + _roi.height / 2.0f;

	//matDump(0, _tmpl, _tmpl->width*_tmpl->height/4);

	features = getFeatures(image, 0, 1.0f, _roi, NULL);
	res = detect(*_tmpl, *features, &peak_value, NULL);

	opt_peak_value = peak_value;
	if(opt_peak_value < gDynamicParam.occlusion_thred){//lost target,and then search 
	
		rcResult = _trkserchMP(image);

		if(_bPredict){
			if(utcStatus == 1)
				_bTrkSchLost = false;
			if(utcStatus == 0 && _bTrkSchLost == false){
				_bTrkSchLost = true;
				_trkSchLstRc = rcResult;
			}
			if(utcStatus == 0 && _bTrkSchLost == true){
				centPt.x = _trkSchLstRc.x + _trkSchLstRc.width/2;
				centPt.y = _trkSchLstRc.y + _trkSchLstRc.height/2;
				if(fabs(centPt.x-pUtcTrkObj->axisX)> _iSegPixel.x){
					_trkSchLstRc.x = (centPt.x>pUtcTrkObj->axisX)?(_trkSchLstRc.x - _iMvPixel2.x):(_trkSchLstRc.x + _iMvPixel2.x);
					utcStatus = 1;
				}else if(fabs(centPt.x-pUtcTrkObj->axisX)> _trkSchLstRc.width/2){
					_trkSchLstRc.x = (centPt.x>pUtcTrkObj->axisX)?(_trkSchLstRc.x - _iMvPixel.x):(_trkSchLstRc.x + _iMvPixel.x);
					utcStatus = 1;
				}else{
					_trkSchLstRc.x = pUtcTrkObj->axisX - _trkSchLstRc.width/2;
				}
				if(fabs(centPt.y-pUtcTrkObj->axisY)> _iSegPixel.y){
					_trkSchLstRc.y = (centPt.y>pUtcTrkObj->axisY)?(_trkSchLstRc.y - _iMvPixel2.y):(_trkSchLstRc.y + _iMvPixel2.y);
					utcStatus = 1;
				}else if(fabs(centPt.y-pUtcTrkObj->axisY)> _trkSchLstRc.height/2){
					_trkSchLstRc.y = (centPt.y>pUtcTrkObj->axisY)?(_trkSchLstRc.y - _iMvPixel.y):(_trkSchLstRc.y + _iMvPixel.y);
					utcStatus = 1;
				}else{
					_trkSchLstRc.y = pUtcTrkObj->axisY - _trkSchLstRc.height/2;
				}
				_roi = _trkSchLstRc;
				if (_roi.x < 0) _roi.x = 0;
				if (_roi.y < 0) _roi.y = 0;
				if (_roi.x + _roi.width > image->width-1)
					_roi.x = image->width -1 - _roi.width;
				if (_roi.y + _roi.height > image->height -1)
					_roi.y = image->height -1 - _roi.height;
			}
		}
		
		_scaleBK = _scale;
		_roiBK = _roi;
	}else{
		// Adjust by cell size and _scale

		_roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
		_roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

		if (_roi.x >= image->width - 1) _roi.x = image->width - 1;
		if (_roi.y >= image->height - 1) _roi.y = image->height - 1;
		if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
		if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

		UTILS_assert(_roi.width >= 0 && _roi.height >= 0);

		features = getFeatures(image, 0, 1.0f, _roi, NULL);
		trainCR(*features, *_prob, _tmpl, _alphaf,interp_factor, lambda, size_patch, true, sigma);

		_scaleBK = _scale;
		_roiBK = _roi;
		
		_feature_peak[featIndx ] = opt_peak_value;
		matCopy(_feature_tmpl[featIndx], _tmpl);
		matCopy(_feature_alphaf[featIndx], _alphaf);
		featIndx++;
		if(featIndx == SAVE_NUM)
			featIndx = 0;

		lostFrame = 0;
		utcStatus  = 1;
		
	}

	return _roi;
}

UTC_RECT_float UtcTrkSrch(UTCTRACK_OBJ *pUtcTrkObj, IMG_MAT frame, int *pRtnStat)
{
	UTC_RECT_float rcResult;
	static UInt32 iCnt = 0;
	UTILS_assert(frame.width == frame.step[0]);
	
	rcResult = _trkseach(pUtcTrkObj, &frame);

	if(utcStatus == 0 && lostFrame > 1){
		updataROI();
//		intervalFrame = gIntervalFrame;
	}
	
	_offsetX = (int)(rcResult.x+rcResult.width/2.f) - pUtcTrkObj->axisX;
	_offsetY = pUtcTrkObj->axisY - (int)(rcResult.y+rcResult.height/2.f);

	if(utcStatus == 0)	// 0-lost 1-trk
		*pRtnStat = 2;	// assi
	else
		*pRtnStat = 1;	// trk

	return rcResult;
}

static bool ConRegAanalyze(UTC_Rect acqWin, UTC_Rect  srRect, Pattern  *pPatterns, int	Num, UTC_Rect *pAcqRect, UTC_Rect *pAcqCentRect)
{
	int i, szIdx = 0, distIdx = 0;
	Pattern	maxSizePat, minDistPat, acqPat;
	int sz, dist, maxSZ = 0, minDist = srRect.width*srRect.height, minDistSZ, maxDist;
	PointICR		acqCent, patCent;
	UTC_SIZE	patSize;
	float ratio = acqWin.width*1.0/acqWin.height;
	float patRatio = 1.0;
	assert(gConRegHdl != NULL && pPatterns != NULL);
	acqCent.x = acqWin.x + acqWin.width/2;
	acqCent.y = acqWin.y + acqWin.height/2;

	if(Num == 0){
		*pAcqRect = acqWin;
		*pAcqCentRect = acqWin;
		return false;
	}else if(Num == 1){
		if(pAcqRect != NULL){
			patCent.x = (pPatterns[0].rightbottom.x + pPatterns[0].lefttop.x)/2;
			patCent.y = (pPatterns[0].rightbottom.y + pPatterns[0].lefttop.y)/2;
			patSize.width = pPatterns[0].rightbottom.x - pPatterns[0].lefttop.x;
			patSize.height = pPatterns[0].rightbottom.y - pPatterns[0].lefttop.y;
			patRatio = patSize.width*1.0/patSize.height;
			if(_bDynamicRatio){
				if(patRatio>1.25){
					ratio = 1.333333;
				}else{
					ratio = 1.0;
				}
			}
//////////////////////////////////////////////////////////////////////////////////////
			pAcqCentRect->width = patSize.width;
			pAcqCentRect->height = patSize.height;
			if(pAcqCentRect->width > pAcqCentRect->height){
				pAcqCentRect->height =pAcqCentRect->width/ratio;
			}else{
				pAcqCentRect->width =pAcqCentRect->height*ratio;
			}
			pAcqCentRect->x	 = patCent.x - pAcqCentRect->width/2;
			pAcqCentRect->y	 = patCent.y - pAcqCentRect->height/2;
///////////////////////////////////////////////////////////////////////////////////////
			pAcqRect->width =(abs(patCent.x-acqCent.x))*2+patSize.width;
			pAcqRect->height = (abs(patCent.y-acqCent.y))*2+patSize.height;
			if(pAcqRect->width > pAcqRect->height){
				pAcqRect->height =pAcqRect->width/ratio;
			}else{
				pAcqRect->width =pAcqRect->height*ratio;
			}
			pAcqRect->x = acqCent.x - pAcqRect->width/2;
			pAcqRect->y = acqCent.y - pAcqRect->height/2;
		}
		return true;
	}

	for(i=0; i<Num; i++){
		patCent.x = (pPatterns[i].rightbottom.x+pPatterns[i].lefttop.x)/2;
		patCent.y = (pPatterns[i].rightbottom.y+pPatterns[i].lefttop.y)/2;
		sz = (pPatterns[i].rightbottom.x-pPatterns[i].lefttop.x)*(pPatterns[i].rightbottom.x-pPatterns[i].lefttop.x);
		dist = (acqCent.x-patCent.x)*(acqCent.x-patCent.x)+(acqCent.y-patCent.y)*(acqCent.y-patCent.y);
		if(sz>maxSZ){	maxSZ = sz;	szIdx = i; maxDist = dist;}
		if(dist<minDist){minDist = dist;	distIdx = i;	minDistSZ = sz;}
	}
	maxSizePat = pPatterns[szIdx];
	minDistPat = pPatterns[distIdx];
	if(maxSZ > minDistSZ*4){
		acqPat = maxSizePat;
	}else{
		acqPat = minDistPat;
	}
	acqPat = minDistPat;
	if(pAcqRect != NULL){
		patCent.x = (acqPat.rightbottom.x + acqPat.lefttop.x)/2;
		patCent.y = (acqPat.rightbottom.y + acqPat.lefttop.y)/2;
		patSize.width = acqPat.rightbottom.x - acqPat.lefttop.x;
		patSize.height = acqPat.rightbottom.y - acqPat.lefttop.y;
		patRatio = patSize.width*1.0/patSize.height;
		if(_bDynamicRatio){
			if(patRatio>1.25){
				ratio = 1.333333;
			}else{
				ratio = 1.0;
			}
		}
		//////////////////////////////////////////////////////////////////////////////////////
		pAcqRect->width =(abs(patCent.x-acqCent.x))*2+patSize.width;
		pAcqRect->height = (abs(patCent.y-acqCent.y))*2+patSize.height;
		if(pAcqRect->width > pAcqRect->height){
			pAcqRect->height =pAcqRect->width/ratio;
		}else{
			pAcqRect->width =pAcqRect->height*ratio;
		}
		pAcqRect->x = acqCent.x - pAcqRect->width/2;
		pAcqRect->y = acqCent.y - pAcqRect->height/2;
//////////////////////////////////////////////////////////////////////////////////////
		pAcqCentRect->width = patSize.width;
		pAcqCentRect->height = patSize.height;
		if(pAcqCentRect->width > pAcqCentRect->height){
			pAcqCentRect->height =pAcqCentRect->width/ratio;
		}else{
			pAcqCentRect->width =pAcqCentRect->height*ratio;
		}
		pAcqCentRect->x	 = patCent.x - pAcqCentRect->width/2;
		pAcqCentRect->y	 = patCent.y - pAcqCentRect->height/2;
///////////////////////////////////////////////////////////////////////////////////////
	}
	return true;
}
static void GetSalientTgt(cv::Mat salientMat[], cv::Rect srRect[], UTC_Rect rcWin, UTC_Rect *pAcqRect, UTC_Rect *pAcqCentRect)
{
	Pattern  pTmpPatt[3][SAMPLE_NUMBER];
	int			patternNum[3];
	cv::Rect 	roiRect[3];
	IMG_MAT	salientMap[3];
	UTC_Rect	salientRect[3];
	UTC_Rect	acqRect[3], blobRect[3];
	int k;
	int iRtn[3]={0,};
	for(k=0; k<3; k++){
		roiRect[k].width = (rcWin.width<srRect[k].width)?rcWin.width:(srRect[k].width-4);
		roiRect[k].height = (rcWin.height<srRect[k].height)?rcWin.height:(srRect[k].height-4);
		roiRect[k].x = (rcWin.width<srRect[k].width)?(rcWin.x-srRect[k].x):2;
		roiRect[k].y =  (rcWin.height<srRect[k].height)?(rcWin.y-srRect[k].y):2;
	}

	for(k=0; k<3; k++){
		salientMap[k].channels = 1;							salientMap[k].dtype = 0;									salientMap[k].data_u8 = salientMat[k].data;
		salientMap[k].width = salientMat[0].cols; 	salientMap[k].height = salientMat[k].rows; salientMap[k].step[0] = salientMat[k].step;
		salientMap[k].size = salientMap[k].width*salientMap[k].height;

		salientRect[k].x = roiRect[k].x;					salientRect[k].y = roiRect[k].y;
		salientRect[k].width = roiRect[k].width;	salientRect[k].height = roiRect[k].height;
		iRtn[k] = GetMoveDetect(gConRegHdl, salientMap[k], salientRect[k], _salientThred, _salientScatter);
		if(iRtn[k]){
			Pattern	*pPAT;
			int	i, *pPatNum;
			pPAT = (Pattern	*)(pTmpPatt[k]);
			pPatNum = &(patternNum[k]);

			for(i=0; i<gConRegHdl->m_patternnum; i++){
				pPAT[i].lefttop.x = gConRegHdl->m_pPatterns[i].lefttop.x + srRect[k].x;
				pPAT[i].lefttop.y = gConRegHdl->m_pPatterns[i].lefttop.y + srRect[k].y;
				pPAT[i].rightbottom.x = gConRegHdl->m_pPatterns[i].rightbottom.x + srRect[k].x;
				pPAT[i].rightbottom.y = gConRegHdl->m_pPatterns[i].rightbottom.y + srRect[k].y;
			}
			*pPatNum = gConRegHdl->m_patternnum;
			salientRect[k].x = srRect[k].x;					salientRect[k].y = srRect[k].y;
			salientRect[k].width = srRect[k].width;	salientRect[k].height = srRect[k].height;
			iRtn[k] = ConRegAanalyze(rcWin, salientRect[k], pPAT, *pPatNum, &acqRect[k], &blobRect[k]);
		}
	}
	int maxIdx=0, sz, minVal = rcWin.width*rcWin.height*2;
	for(k=0; k<3; k++){
		sz = acqRect[k].width*acqRect[k].height;
		if(sz<minVal){
			minVal = sz;
			maxIdx = k;
		}
	}
	if(pAcqRect != NULL){
		pAcqRect->width	= (int)(acqRect[maxIdx].width*_acqRatio);
		pAcqRect->height	= (int)(acqRect[maxIdx].height*_acqRatio);
		pAcqRect->width = (pAcqRect->width<rcWin.width)?pAcqRect->width:rcWin.width;
		pAcqRect->height = (pAcqRect->height<rcWin.height)?pAcqRect->height:rcWin.height;
		pAcqRect->x = (int)((acqRect[maxIdx].x+acqRect[maxIdx].width/2.0)-pAcqRect->width/2.0);
		pAcqRect->y = (int)((acqRect[maxIdx].y+acqRect[maxIdx].height/2.0)-pAcqRect->height/2.0);
	}
	if(pAcqCentRect != NULL){
		*pAcqCentRect	= blobRect[maxIdx];
	}
}

#define		SCALE_NUM	3
bool UtcTrkPreAcq(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param acqParam, UTC_Rect *pAcqRect, UTC_Rect *pAcqCentRect, void  *pPatterns, int	*pNum)
{
	int i, j, k;
	bool iRtn = 0;
	UTC_Rect	acqRect, blobRect;
	Pattern  pTmpPatt[SAMPLE_NUMBER];
	int			patternNum;
	if(gConRegHdl == NULL){
		if(pAcqRect != NULL){
			*pAcqRect	= acqParam.rcWin;
		}
		if(pAcqCentRect != NULL){
			*pAcqCentRect	= acqParam.rcWin;
		}
		return false;
	}
	IMG_MAT	salientMap;
	UTC_Rect	salientRect;
	cv::Mat srMat[SCALE_NUM] , salientMat[SCALE_NUM];
	cv::Rect  srRect[SCALE_NUM], imgRect[SCALE_NUM], roiRect[SCALE_NUM];
	cv::Mat image = cv::Mat(frame.height, frame.width, CV_8UC1, frame.data_u8);
	int scalerNum = (_bmultScaler==1)?3:1;

#pragma omp parallel for
	for(k=0; k<scalerNum; k++){
		int jj, scalerValue;
		if(k==0)
			scalerValue = _largeScaler;
		else if(k==1)
			scalerValue = _midScaler;
		else if(k==2)
			scalerValue = _smallScaler;
		srRect[k].width = scalerValue;
		srRect[k].height = scalerValue;
		srRect[k].x = (acqParam.rcWin.x+acqParam.rcWin.width/2) - srRect[k].width/2;
		srRect[k].y = (acqParam.rcWin.y+acqParam.rcWin.height/2) - srRect[k].height/2;

		imgRect[k].x = imgRect[k]. y= 0;
		imgRect[k].width = frame.width;
		imgRect[k].height = frame.height;
		_overlapRoi(srRect[k], imgRect[k], &roiRect[k]);
		srRect[k] = roiRect[k];

		assert(srRect[k].width !=0 );
		assert(srRect[k].height !=0 );
		image(srRect[k]).copyTo(srMat[k]);
	}

#pragma omp parallel for
	for(k=0; k<scalerNum; k++){
		cv::Size kelsz = cv::Size(64, 64);
		salientMat[k] = GetSR(srMat[k], kelsz);
	}

	UInt8 *pSR[SCALE_NUM];
	for(k=1; k<scalerNum; k++){
		pSR[0] = (UInt8 *)salientMat[0].data+(srRect[k].y-srRect[0].y)*salientMat[0].step + (srRect[k].x-srRect[0].x);
		pSR[k] = (UInt8 *)salientMat[k].data;
		for(j=0; j<srRect[k].height; j++){
			for(i=0; i<srRect[k].width; i++){
				*(pSR[0]+j*salientMat[0].step+i) = (*(pSR[k]+j*salientMat[k].step+i) > *(pSR[0]+j*salientMat[0].step+i))?(*(pSR[k]+j*salientMat[k].step+i)):(*(pSR[0]+j*salientMat[0].step+i));
			}
		}
	}

	roiRect[0].width = (acqParam.rcWin.width<srRect[0].width)?acqParam.rcWin.width:(srRect[0].width-4);
	roiRect[0].height = (acqParam.rcWin.height<srRect[0].height)?acqParam.rcWin.height:(srRect[0].height-4);
	roiRect[0].x = (acqParam.rcWin.width<srRect[0].width)?(acqParam.rcWin.x-srRect[0].x):2;
	roiRect[0].y =  (acqParam.rcWin.height<srRect[0].height)?(acqParam.rcWin.y-srRect[0].y):2;

	salientMap.channels = 1;	salientMap.dtype = 0;	salientMap.data_u8 = salientMat[0].data;
	salientMap.width = salientMat[0].cols; salientMap.height = salientMat[0].rows; salientMap.step[0] = salientMat[0].step;
	salientMap.size = salientMap.width*salientMap.height;
	salientRect.x = roiRect[0].x;	salientRect.y = roiRect[0].y;
	salientRect.width = roiRect[0].width;	salientRect.height = roiRect[0].height;
	iRtn = GetMoveDetect(gConRegHdl, salientMap, salientRect, _salientThred, _salientScatter);
	if(iRtn){
		Pattern	*pPAT;
		int	*pPatNum;
		if(pPatterns != NULL && pNum!= NULL){
			pPAT = (Pattern	*)pPatterns;
			pPatNum = pNum;
		}else{
			pPAT = (Pattern	*)pTmpPatt;
			pPatNum = &patternNum;
		}
		for(i=0; i<gConRegHdl->m_patternnum; i++){
			pPAT[i].lefttop.x = gConRegHdl->m_pPatterns[i].lefttop.x + srRect[0].x;
			pPAT[i].lefttop.y = gConRegHdl->m_pPatterns[i].lefttop.y + srRect[0].y;
			pPAT[i].rightbottom.x = gConRegHdl->m_pPatterns[i].rightbottom.x + srRect[0].x;
			pPAT[i].rightbottom.y = gConRegHdl->m_pPatterns[i].rightbottom.y + srRect[0].y;
		}
		*pPatNum = gConRegHdl->m_patternnum;
		salientRect.x = srRect[0].x;	salientRect.y = srRect[0].y;
		salientRect.width = srRect[0].width;	salientRect.height = srRect[0].height;
		iRtn = ConRegAanalyze(acqParam.rcWin, salientRect, pPAT, *pPatNum, &acqRect, &blobRect);
		if(pAcqRect != NULL){
			pAcqRect->width	= (int)(acqRect.width*_acqRatio);
			pAcqRect->height	= (int)(acqRect.height*_acqRatio);
			pAcqRect->width = (pAcqRect->width<acqParam.rcWin.width)?pAcqRect->width:acqParam.rcWin.width;
			pAcqRect->height = (pAcqRect->height<acqParam.rcWin.height)?pAcqRect->height:acqParam.rcWin.height;
			pAcqRect->x = (int)((acqRect.x+acqRect.width/2.0)-pAcqRect->width/2.0);
			pAcqRect->y = (int)((acqRect.y+acqRect.height/2.0)-pAcqRect->height/2.0);
		}
		if(pAcqCentRect != NULL){
			*pAcqCentRect	= blobRect;
		}
	}
	if(!iRtn ){
		if(pAcqRect != NULL){
			*pAcqRect	= acqParam.rcWin;
		}
		if(pAcqCentRect != NULL){
			*pAcqCentRect	= acqParam.rcWin;
		}
	}

	return iRtn;
}

bool UtcTrkPreAcqSR(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param acqParam, UTC_Rect *pAcqRect)
{
	Pattern  pPatterns[SAMPLE_NUMBER];
	int			patternNum;

	return UtcTrkPreAcq(pUtcTrkObj, frame, acqParam, pAcqRect, NULL, (void*)pPatterns, &patternNum);
}

bool UtcBlobDetectSR(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param acqParam, UTC_Rect *pBlobRect)
{
	Pattern  pPatterns[SAMPLE_NUMBER];
	int			patternNum;

	return UtcTrkPreAcq(pUtcTrkObj, frame, acqParam, NULL, pBlobRect, (void*)pPatterns, &patternNum);
}

bool UtcTrkPreAcq2(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param acqParam, UTC_Rect *pAcqRect, UTC_Rect *pAcqCentRect,
									UInt8*srMap, UInt8*sobelMap, void  *pPatterns, int	*pNum)
{
	int i, j, k;
	bool iRtn = 0;
	UTC_Rect	acqRect, blobRect, roiRc;
	int SSIM;
	IMG_MAT_UCHAR	tmpSrc, tmpMat;
	Pattern  pTmpPatt[SAMPLE_NUMBER];
	int			patternNum;

	if(gConRegHdl == NULL){
		if(pAcqRect != NULL){
			*pAcqRect	= acqParam.rcWin;
		}
		if(pAcqCentRect != NULL){
			*pAcqCentRect	= acqParam.rcWin;
		}
		return false;
	}
	IMG_MAT	salientMap;
	UTC_Rect	salientRect;
	cv::Mat srMat[SCALE_NUM] , salientMat[SCALE_NUM];
	cv::Rect  srRect[SCALE_NUM], imgRect[SCALE_NUM], roiRect[SCALE_NUM];
	cv::Mat image = cv::Mat(frame.height, frame.width, CV_8UC1, frame.data_u8);
	int scalerNum = (_bmultScaler==1)?3:1;

#pragma omp parallel for
	for(k=0; k<scalerNum; k++){
			int jj, scalerValue;
			if(k==0)
				scalerValue = _largeScaler;
			else if(k==1)
				scalerValue = _midScaler;
			else if(k==2)
				scalerValue = _smallScaler;
			srRect[k].width = scalerValue;
			srRect[k].height = scalerValue;
			srRect[k].x = (acqParam.rcWin.x+acqParam.rcWin.width/2) - srRect[k].width/2;
			srRect[k].y = (acqParam.rcWin.y+acqParam.rcWin.height/2) - srRect[k].height/2;

			imgRect[k].x = imgRect[k]. y= 0;
			imgRect[k].width = frame.width;
			imgRect[k].height = frame.height;
			_overlapRoi(srRect[k], imgRect[k], &roiRect[k]);
			srRect[k] = roiRect[k];

			assert(srRect[k].width !=0 );
			assert(srRect[k].height !=0 );
			image(srRect[k]).copyTo(srMat[k]);
		}

	if(sobelMap != NULL){
		tmpMat.dtype = 0;							tmpMat.channels = 1;
		tmpMat.width = _largeScaler;		tmpMat.height = _largeScaler;
		tmpMat.data_u8 = sobelMap;		tmpMat.step[0] = tmpMat.width;
		tmpSrc.dtype = 0;								tmpSrc.channels = 1;
		tmpSrc.width = _largeScaler;			tmpSrc.height = _largeScaler;
		tmpSrc.data_u8 = srMat[0].data;		tmpSrc.step[0] = tmpMat.width;
		_IMG_sobel(tmpSrc, &tmpMat);
	}

#pragma omp parallel for
	for(k=0; k<scalerNum; k++){
		cv::Size kelsz = cv::Size(64, 64);
		salientMat[k] = GetSR(srMat[k], kelsz);
	}

	UInt8 *pSR[SCALE_NUM];
	for(k=1; k<scalerNum; k++){
		pSR[0] = (UInt8 *)salientMat[0].data+(srRect[k].y-srRect[0].y)*salientMat[0].step + (srRect[k].x-srRect[0].x);
		pSR[k] = (UInt8 *)salientMat[k].data;
		for(j=0; j<srRect[k].height; j++){
			for(i=0; i<srRect[k].width; i++){
				*(pSR[0]+j*salientMat[0].step+i) = (*(pSR[k]+j*salientMat[k].step+i) > *(pSR[0]+j*salientMat[0].step+i))?(*(pSR[k]+j*salientMat[k].step+i)):(*(pSR[0]+j*salientMat[0].step+i));
			}
		}
	}
	if(srMap != NULL){
		memcpy(srMap, salientMat[0].data, salientMat[0].step*salientMat[0].rows);
	}

	roiRect[0].width = (acqParam.rcWin.width<srRect[0].width)?acqParam.rcWin.width:(srRect[0].width-4);
	roiRect[0].height = (acqParam.rcWin.height<srRect[0].height)?acqParam.rcWin.height:(srRect[0].height-4);
	roiRect[0].x = (acqParam.rcWin.width<srRect[0].width)?(acqParam.rcWin.x-srRect[0].x):2;
	roiRect[0].y =  (acqParam.rcWin.height<srRect[0].height)?(acqParam.rcWin.y-srRect[0].y):2;

	salientMap.channels = 1;	salientMap.dtype = 0;	salientMap.data_u8 = salientMat[0].data;
	salientMap.width = salientMat[0].cols; salientMap.height = salientMat[0].rows; salientMap.step[0] = salientMat[0].step;
	salientMap.size = salientMap.width*salientMap.height;
	salientRect.x = roiRect[0].x;	salientRect.y = roiRect[0].y;
	salientRect.width = roiRect[0].width;	salientRect.height = roiRect[0].height;
	iRtn = GetMoveDetect(gConRegHdl, salientMap, salientRect, _salientThred, _salientScatter);
	if(iRtn){
		Pattern	*pPAT;
		int	*pPatNum;
		if(pPatterns != NULL && pNum!= NULL){
			pPAT = (Pattern	*)pPatterns;
			pPatNum = pNum;
		}else{
			pPAT = (Pattern	*)pTmpPatt;
			pPatNum = &patternNum;
		}
		for(i=0; i<gConRegHdl->m_patternnum; i++){
			pPAT[i].lefttop.x = gConRegHdl->m_pPatterns[i].lefttop.x + srRect[0].x;
			pPAT[i].lefttop.y = gConRegHdl->m_pPatterns[i].lefttop.y + srRect[0].y;
			pPAT[i].rightbottom.x = gConRegHdl->m_pPatterns[i].rightbottom.x + srRect[0].x;
			pPAT[i].rightbottom.y = gConRegHdl->m_pPatterns[i].rightbottom.y + srRect[0].y;
		}
		*pPatNum = gConRegHdl->m_patternnum;
		salientRect.x = srRect[0].x;	salientRect.y = srRect[0].y;
		salientRect.width = srRect[0].width;	salientRect.height = srRect[0].height;
		iRtn = ConRegAanalyze(acqParam.rcWin, salientRect, pPAT, *pPatNum, &acqRect, &blobRect);
		if(pAcqRect != NULL){
			pAcqRect->width	= (int)(acqRect.width*_acqRatio);
			pAcqRect->height	= (int)(acqRect.height*_acqRatio);
			pAcqRect->width = (pAcqRect->width<acqParam.rcWin.width)?pAcqRect->width:acqParam.rcWin.width;
			pAcqRect->height = (pAcqRect->height<acqParam.rcWin.height)?pAcqRect->height:acqParam.rcWin.height;
			pAcqRect->x = (int)((acqRect.x+acqRect.width/2.0)-pAcqRect->width/2.0);
			pAcqRect->y = (int)((acqRect.y+acqRect.height/2.0)-pAcqRect->height/2.0);
		}
		if(pAcqCentRect != NULL){
			*pAcqCentRect	= blobRect;
		}
	}

	if(!iRtn ){
		if(pAcqRect != NULL){
			*pAcqRect	= acqParam.rcWin;
		}
		if(pAcqCentRect != NULL){
			*pAcqCentRect	= acqParam.rcWin;
		}
	}

	return iRtn;
}

UTC_RECT_float UtcTrkAcqSR(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param inputParam, bool bSalient)
{
	UTC_RECT_float rcInit;
	UTC_Rect AcqRect;
	Pattern  pPatterns[SAMPLE_NUMBER];
	int			patternNum;

	pUtcTrkObj->axisX = inputParam.axisX;
	pUtcTrkObj->axisY = inputParam.axisY;
	rcInit.x = inputParam.rcWin.x;
	rcInit.y = inputParam.rcWin.y;
	rcInit.width = inputParam.rcWin.width;
	rcInit.height = inputParam.rcWin.height;

	UTILS_assert(frame.width == frame.step[0]);

	unInit();

	if(bSalient){
		UtcTrkPreAcq(pUtcTrkObj, frame, inputParam, &AcqRect, NULL,	(void*)pPatterns, &patternNum);
		rcInit.width = (AcqRect.width<_minAcqSize.width)?_minAcqSize.width:AcqRect.width;
		rcInit.height = (AcqRect.height<_minAcqSize.height)?_minAcqSize.height:AcqRect.height;
		rcInit.x = (AcqRect.x+AcqRect.width/2.0)-rcInit.width/2;
		rcInit.y = (AcqRect.y+AcqRect.height/2.0)-rcInit.height/2;
	}

	if(_bBlurFilter)
		blurFilterROI(&frame, rcInit);

	if(_bCalSceneMV || _bSceneMVRecord){
		_unInitSceneMV(gSceneMVHdl);

		if(_bEnhScene){
			getSceneMapEnh(gSceneMVHdl, &frame, 1, _nrxScene, _nryScene, _fCliplimit);
		}else{
			getSceneMap(gSceneMVHdl, &frame, 1);
		}
		_sceneFrmCount = 0;
		_sceneFrmGap = 1;
		_sceneFrmSum = 0;
		clrTrkState(gSceneMVHdl);
	}
	if(_bKalmanFilter && _bCalSceneMV){
		PointfCR		trkPoint;
		trkPoint.x = 0.f;
		trkPoint.y = 0.f;
		gKalmanTrkObj.KalmanTrkAcq(trkPoint,_KalmanCoefQ, _KalmanCoefR);
	}

	if(_bEnhROI)
		enhROI(&frame, rcInit, 2, 2, _fCliplimit);

	init(rcInit, &frame);

	_offsetX = (int)(rcInit.x+rcInit.width/2.f) - pUtcTrkObj->axisX;
	_offsetY = pUtcTrkObj->axisY - (int)(rcInit.y+rcInit.height/2.f);
	_trkSchAcqRc = rcInit;
	_bTrkSchLost = false;

	if((abs(_offsetX)> resTrackObj.res_distance ||  abs(_offsetY)> resTrackObj.res_distance) &&
		rcInit.width*rcInit.height < resTrackObj.res_area)
		_workMode = _work_mode_serch;
	else
		_workMode = _work_mode_trk;

	if(frame.width>1280){
		if(min_bias_pix	== 200 && max_bias_pix == 600 ){
			min_bias_pix	=	260;
			max_bias_pix	= 700;
			_blendValue =  100;
		}
	}else if(frame.width>768 && frame.width<=1280){
		if(min_bias_pix	== 200 && max_bias_pix == 600 ){
			min_bias_pix	=	80;
			max_bias_pix	= 400;
			_blendValue =  50;
		}
	}else if(frame.width<=768){
		if(min_bias_pix	== 200 && max_bias_pix == 600 ){
			min_bias_pix	=	40;
			max_bias_pix	= 200;
			_blendValue =  25;
		}
	}

	_bTrackModel = TRACK_MODEL_STATUS;

	_bStartBackTrk = false;
	startFrames = 0;
	trkPosNmu = 0;

	continueCST = 0;
	continuePeak = 0;
	_bSimStatus = SEARCH_SIM_STATUS;

	if(_bBackTrack && gHistTrkHdl){
		SetReBackFlag(gHistTrkHdl);
	}

	return rcInit;
}

UTC_RECT_float UtcTrkAcq(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param inputParam)
{
	UTC_RECT_float rcInit;

	pUtcTrkObj->axisX = inputParam.axisX;
	pUtcTrkObj->axisY = inputParam.axisY;
	rcInit.x = inputParam.rcWin.x;
	rcInit.y = inputParam.rcWin.y;
	rcInit.width = inputParam.rcWin.width;
	rcInit.height = inputParam.rcWin.height;

	UTILS_assert(frame.width == frame.step[0]);

	unInit();

	if(_bBlurFilter)
		blurFilterROI(&frame, rcInit);

	if(_bCalSceneMV || _bSceneMVRecord){
		_unInitSceneMV(gSceneMVHdl);
		if(_bEnhScene){
			getSceneMapEnh(gSceneMVHdl, &frame, 1,  _nrxScene, _nryScene,  _fCliplimit);
		}else{
			getSceneMap(gSceneMVHdl, &frame, 1);
		}
		_sceneFrmCount = 0;
		_sceneFrmGap = 1;
		_sceneFrmSum = 0;
		clrTrkState(gSceneMVHdl);
	}
	if(_bKalmanFilter && _bCalSceneMV){
		PointfCR		trkPoint;
		trkPoint.x = 0.f;
		trkPoint.y = 0.f;
		gKalmanTrkObj.KalmanTrkAcq(trkPoint, _KalmanCoefQ, _KalmanCoefR);
	}

	if(_bEnhROI)
		enhROI(&frame, rcInit, 2, 2, _fCliplimit);

	init(rcInit, &frame);

	_offsetX = (int)(rcInit.x+rcInit.width/2.f) - pUtcTrkObj->axisX;
	_offsetY = pUtcTrkObj->axisY - (int)(rcInit.y+rcInit.height/2.f);
	_trkSchAcqRc = rcInit;
	_bTrkSchLost = false;

	if((abs(_offsetX)> resTrackObj.res_distance ||  abs(_offsetY)> resTrackObj.res_distance) &&
		rcInit.width*rcInit.height < resTrackObj.res_area)
		_workMode = _work_mode_serch;
	else
		_workMode = _work_mode_trk;

	if(frame.width>1280){
		if(min_bias_pix	== 200 && max_bias_pix == 600 ){
			min_bias_pix	=	260;
			max_bias_pix	= 700;
			_blendValue =  100;
		}
	}else if(frame.width>768 && frame.width<=1280){
		if(min_bias_pix	== 200 && max_bias_pix == 600 ){
			min_bias_pix	=	80;
			max_bias_pix	= 400;
			_blendValue =  50;
		}
	}else if(frame.width<=768){
		if(min_bias_pix	== 200 && max_bias_pix == 600 ){
			min_bias_pix	=	40;
			max_bias_pix	= 200;
			_blendValue =  25;
		}
	}

	_bTrackModel = TRACK_MODEL_STATUS;

	_bStartBackTrk = false;
	startFrames = 0;
	trkPosNmu = 0;

	continueCST = 0;
	continuePeak = 0;
	_bSimStatus = SEARCH_SIM_STATUS;

	if(_bBackTrack && gHistTrkHdl){
		SetReBackFlag(gHistTrkHdl);
	}

	return rcInit;
}

static bool	JudgeBackTrk(UTC_RECT_float	trkRC, UTCTRACK_OBJ *pUtcTrkObj, int continueFrms, PointfCR histVelocity)
{
	float	dist, speed, totalspd, cmpspd,diag;
	PointfCR		axisPt, trkPt;
	axisPt.x = pUtcTrkObj->axisX;
	axisPt.y = pUtcTrkObj->axisY;
	trkPt.x = trkRC.x + trkRC.width/2;
	trkPt.y = trkRC.y + trkRC.height/2;
	diag = sqrt(trkRC.width*trkRC.width+trkRC.height*trkRC.height);
	dist = sqrt((axisPt.x-trkPt.x)*(axisPt.x-trkPt.x)+(axisPt.y-trkPt.y)*(axisPt.y-trkPt.y));
	speed = sqrt(histVelocity.x*histVelocity.x+histVelocity.y*histVelocity.y);
	totalspd = continueFrms*speed;
	cmpspd = totalspd*_fRatioFrms;
	cmpspd = (cmpspd<diag)?diag:cmpspd;
	if(dist<cmpspd){
		return true;
	}else{
		return false;
	}
}

static float bbOverlap(const UTC_RECT_float box1,const UTC_RECT_float box2)
{
	float colInt , rowInt, intersection, area1,area2;
	if (box1.x > box2.x+box2.width) { return 0.0; }
	if (box1.y > box2.y+box2.height) { return 0.0; }
	if (box1.x+box1.width < box2.x) { return 0.0; }
	if (box1.y+box1.height < box2.y) { return 0.0; }

	colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
	rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

	intersection = colInt * rowInt;
	area1 = box1.width*box1.height;
	area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

static void DetectSimilar(void *pObj, TRAJECTORY_PARAM	*pTrajParam, IMG_MAT *image, UTC_RECT_float trkRoi, IMG_MAT *tmpl, int type)
{
	int	k;
	float peak_value[25],max_value = 0.0;
	float _overlap;
	PointfCR res[25];
	UTC_RECT_float  rc, rect[25];
	float	serchAngle, tragjAngle;
	PointfCR		serchPt;

	HIST_TRACK_HANDLE pHistTrkObj;
	SUB_TRK_PARAM	*pSubTrkParam;

	float diagdist = sqrt(trkRoi.width*trkRoi.width+trkRoi.height*trkRoi.height);
	memset(peak_value, 0, sizeof(peak_value));
	if(type == 1){
		pHistTrkObj = (HIST_TRACK_HANDLE)pObj;
		pHistTrkObj->bsim = false;
	}else{
		pSubTrkParam = (SUB_TRK_PARAM	*)pObj;
		pSubTrkParam->bsim = false;
	}

	float	mvThred;
	if(image->width <= 768){
		mvThred = _peakMvThred/2;
	}else if(image->width <= 1280){
		mvThred = _peakMvThred;
	}else{
		mvThred = _peakMvThred*1.5;
	}
	PointfCR histDelta = pTrajParam->m_histDelta;
	float histSqrt = sqrt(histDelta.x*histDelta.x + histDelta.y*histDelta.y);
	if(histSqrt<mvThred){
		return;
	}

//#pragma omp parallel for
	for(k=0; k<2; k++)
	{
		bool  bRun = true;
		if(k==0){
			rect[k].x = trkRoi.x + cos(pTrajParam->m_trajAng*CR_PI/180.0)*diagdist*0.8;
			rect[k].y = trkRoi.y - sin(pTrajParam->m_trajAng*CR_PI/180.0)*diagdist*0.8;
		}else if(k==1){
			rect[k].x = trkRoi.x - cos(pTrajParam->m_trajAng*CR_PI/180.0)*diagdist*0.8;
			rect[k].y = trkRoi.y + sin(pTrajParam->m_trajAng*CR_PI/180.0)*diagdist*0.8;
		}
		rect[k].width = trkRoi.width;
		rect[k].height = trkRoi.height;

		if (rect[k].x + rect[k].width <= 0 || rect[k].y + rect[k].height <= 0 ) {
			peak_value[k] = 0.0;
			bRun = false;
		}
		if (rect[k].x >= image->width - 1 || rect[k].y >= image->height - 1) {
			peak_value[k] = 0.0;
			bRun = false;
		}

		if (rect[k].x < 0) rect[k].x = 0;
		if (rect[k].y < 0) rect[k].y = 0;
		if (rect[k].x + rect[k].width > image->width-1)			rect[k].x = image->width-1 - rect[k].width;
		if (rect[k].y + rect[k].height > image->height-1)		rect[k].y = image->height-1 - rect[k].height;

		if(bRun){
			IMG_MAT *features = NULL;
			features = getFeatures(image, 0, 1.0f, rect[k], &features);
			res[k] = detect(*tmpl, *features, &peak_value[k], NULL);
			matFree(features);
		}
	}

	for(k=0; k<2; k++)
	{
		if(peak_value[k] > max_value && peak_value[k] > gDynamicParam.occlusion_thred)
		{
			rc.x = rect[k].x + ((float) res[k].x * cell_size * _scale);
			rc.y = rect[k].y + ((float) res[k].y * cell_size * _scale);
			rc.width = rect[k].width;
			rc.height = rect[k].height;

			if (rc.x < 0) rc.x = 0;
			if (rc.y < 0) rc.y = 0;
			if (rc.x + rc.width > image->width-1)		rc.x = image->width-1 - rc.width;
			if (rc.y + rc.height > image->height-1)	rc.y = image->height-1 - rc.height;
			assert(rc.width >= 0 && rc.height >= 0);

			serchPt.x = (rc.x+rc.width/2) - pTrkObj->axisX;//axis cordinate
			serchPt.y = pTrkObj->axisY-(rc.y+rc.height/2);
			serchAngle =  _getAngle(serchPt);
			tragjAngle = pTrajParam->m_trajAng;
			if(serchAngle>=0.0 && serchAngle<=90.0 && tragjAngle>=270.0 && tragjAngle<=360.0){
				serchAngle+= 360.0;
			}else if(tragjAngle>=0.0 && tragjAngle<=90.0 && serchAngle>=270.0 && serchAngle<=360.0){
				tragjAngle+= 360.0;
			}
			if(fabs(serchAngle-tragjAngle)<20 || (fabs(serchAngle-tragjAngle) >170 && fabs(serchAngle-tragjAngle) <190 ) ){
				 _overlap = bbOverlap(trkRoi, rc);
				if(_overlap <0.2){
					max_value = peak_value[k];
					if(type == 1){
						pHistTrkObj->bsim = true;
						pHistTrkObj->simRect =rc;
					}else{
						pSubTrkParam->bsim = true;
						pSubTrkParam->simRect =rc;
					}
				}
			}
		}
	}
}

static void TrackSimilar(SUB_TRK_PARAM	*pSubTrkParam, TRAJECTORY_PARAM	*pTrajParam, IMG_MAT *image, UTC_RECT_float trkRoi, IMG_MAT *tmpl)
{
	PointfCR res;
	UTC_RECT_float  rc;
	IMG_MAT *features = NULL;
	features = getFeatures(image, 0, 1.0f, pSubTrkParam->_roi, &features);
	res = detect(*tmpl, *features, &pSubTrkParam->opt_peak_value, NULL);
	matFree(features);
	if(pSubTrkParam->opt_peak_value > gDynamicParam.occlusion_thred)
	{
		rc.x = pSubTrkParam->_roi.x + ((float) res.x * cell_size * _scale);
		rc.y = pSubTrkParam->_roi.y + ((float) res.y * cell_size * _scale);
		rc.width = pSubTrkParam->_roi.width;
		rc.height = pSubTrkParam->_roi.height;

		if (rc.x < 0) rc.x = 0;
		if (rc.y < 0) rc.y = 0;
		if (rc.x + rc.width > image->width-1)		rc.x = image->width-1 - rc.width;
		if (rc.y + rc.height > image->height-1)	rc.y = image->height-1 - rc.height;
		assert(rc.width >= 0 && rc.height >= 0);

		pSubTrkParam->_roi = rc;
		pSubTrkParam->utcStatus = 1;
	}else{
		pSubTrkParam->utcStatus = 0;
	}
}

static void MultiPeakJudge(SUB_TRK_PARAM	*pSubTrkParam, TRAJECTORY_PARAM	*pTrajParam, IMG_MAT *image, IMG_MAT *peakMat, UTC_RECT_float peakRoi, PointfCR peakRes)
{
	int i, j, m, n, k;
	float *pPeak, *pRes;

	int subw = (peakMat->width*0.25);
	int subh = (peakMat->height*0.25);
	int validw, validh;
	float MaxValue, posValue[100];
	PointICR MaxIdx, posIdx[100];
	int posCount = 0;
	pSubTrkParam->bsim = false;

	float		mvThred;
	if(image->width <= 768){
		mvThred = _peakMvThred/2;
	}else if(image->width <= 1280){
		mvThred = _peakMvThred;
	}else{
		mvThred = _peakMvThred*1.5;
	}
	PointfCR histDelta = pTrajParam->m_histDelta;
	float histSqrt = sqrt(histDelta.x*histDelta.x + histDelta.y*histDelta.y);
	if(histSqrt<mvThred){
		return;
	}

	IMG_MAT *resMat = matAlloc(MAT_float, peakMat->width, peakMat->height, 1);
	for(j=0; j<peakMat->height; j+=subh){
		for(i=0; i<peakMat->width; i+=subw){

			pPeak = peakMat->data+j*peakMat->width+i;
			pRes = resMat->data+j*resMat->width+i;

			validw = (i+subw)<peakMat->width?subw:(peakMat->width-i);
			validh = (j+subh)<peakMat->height?subh:(peakMat->height-j);
			MaxValue = 0.f;

			for(m=0; m<validh; m++){
				for(n=0; n<validw; n++){
					if(*(pPeak+m*peakMat->width+n) > MaxValue){
						MaxValue = *(pPeak+m*peakMat->width+n) ;
						MaxIdx.x = n;	MaxIdx.y = m;
					}
				}
			}

			if(MaxValue>gDynamicParam.occlusion_thred){
				*(pRes+MaxIdx.y*resMat->width+MaxIdx.x) = MaxValue;
				posIdx[posCount] = MaxIdx;
				posValue[posCount] = MaxValue;
				posCount++;
			}
		}
	}

	PointfCR res;
	UTC_RECT_float  rc;
	float	serchAngle, tragjAngle;
	PointfCR		serchPt;

	if(posCount >1){
		for(k=0; k<posCount; k++){
			res.x = posIdx[k].x; res.y = posIdx[k].y;
			if(fabs(res.x-peakRes.x)<1 && fabs(res.y-peakRes.y)<1)
				continue;
			rc.x = peakRoi.x + ((float) res.x * cell_size * _scale);
			rc.y = peakRoi.y + ((float) res.y * cell_size * _scale);
			rc.width = peakRoi.width;
			rc.height = peakRoi.height;

			if (rc.x < 0) rc.x = 0;
			if (rc.y < 0) rc.y = 0;
			if (rc.x + rc.width > image->width-1)		rc.x = image->width-1 - rc.width;
			if (rc.y + rc.height > image->height-1)	rc.y = image->height-1 - rc.height;
			assert(rc.width >= 0 && rc.height >= 0);

			serchPt.x = (rc.x+rc.width/2) - pTrkObj->axisX;//axis cordinate
			serchPt.y = pTrkObj->axisY-(rc.y+rc.height/2);
			serchAngle =  _getAngle(serchPt);
			tragjAngle = pTrajParam->m_trajAng;
			if(serchAngle>=0.0 && serchAngle<=90.0 && tragjAngle>=270.0 && tragjAngle<=360.0){
				serchAngle+= 360.0;
			}else if(tragjAngle>=0.0 && tragjAngle<=90.0 && serchAngle>=270.0 && serchAngle<=360.0){
				tragjAngle+= 360.0;
			}
			if(fabs(serchAngle-tragjAngle)<30 || (fabs(serchAngle-tragjAngle) >165 && fabs(serchAngle-tragjAngle) <195 ) ){
				pSubTrkParam->bsim = true;
				pSubTrkParam->simRect =rc;
				break;
			}
		}
	}

	free(resMat);
}

static UInt32 continueCount = 0;

//#define SIM_MODEL_1
//#define SIM_MODEL_2
#define	 SIM_MODEL_3

static void *mainProcTsk(void *ctxHdl)
{
	OSA_printf("%s: enter.", __func__);

	while(gBackThrTsk.exitProcThread ==  false)
	{
		OSA_semWait(&gBackThrTsk.procNotifySem, OSA_TIMEOUT_FOREVER);

		int i;
		float ratio;
		float opt = opt_peak_value;
		UTC_RECT_float roi = _roi;
		int trkStatus = utcStatus;
		IMG_MAT *_thrMap = matAlloc(_tskMap->dtype, _tskMap->width, _tskMap->height, _tskMap->channels);
		matCopy(_thrMap, _tskMap);
		backStatus = 1;

		if(_bCalSceneMV || _bSceneMVRecord){
			_sceneFrmCount++;
			_sceneFrmSum++;

			if(_sceneFrmCount == _sceneFrmGap){

				if(_bEnhScene){
					getSceneMapEnh(gSceneMVHdl, _thrMap, 0,  _nrxScene, _nryScene,  _fCliplimit);
				}else{
					getSceneMap(gSceneMVHdl, _thrMap, 0);
				}
				for(i=0; i<_sceneFrmGap; i++){
					ratio = 1.0/_sceneFrmGap;
					TrackJudge(gSceneMVHdl, ratio);
				}
				_sceneFrmCount = 0;

			}else if(_sceneFrmSum >= 120 && _sceneFrmCount == 1){
				if(_bEnhScene){
					getSceneMapEnh(gSceneMVHdl, _thrMap, 0,  _nrxScene, _nryScene,  _fCliplimit);
				}else{
					getSceneMap(gSceneMVHdl, _thrMap, 0);
				}
				TrackJudge(gSceneMVHdl, 1.0);

				PointfCR		trkPoint;
				TRAJECTORY_PARAM	*pTrajParam = &gSceneMVHdl->m_trajParam;
				trkPoint = pTrajParam->m_trackDelta;
				float absV = sqrt(trkPoint.x*trkPoint.x+trkPoint.y*trkPoint.y);
				if(absV <1e-6){
					_sceneFrmGap = _sceneFrmThred;
				}else if(absV > 4){
					_sceneFrmGap = 1;
				}else if(absV > 2){
					_sceneFrmGap = 2;
				}else if(absV > 1){
					_sceneFrmGap = 4;
				}else if(absV > 1e-6){
					_sceneFrmGap = _sceneFrmThred;
				}
				_sceneFrmCount = 0;
				_sceneFrmSum = 0;
			}
		}

		if(_bBackTrack && utcStatus==1){//track status process
			if(_bTrackModel == TRACK_MODEL_STATUS){
				backStatus = BackTrackStatus(gHistTrkHdl, gSceneMVHdl, *_thrMap, UtcTrk0bj, opt, roi, trkStatus);
				if(backStatus == 0){
					_bTrackModel = BACK_MODEL_STATUS;
					continueCount = 0;
				}
			}else if(_bTrackModel == BACK_MODEL_STATUS){
				continueCount++;
				bool bJudge = JudgeBackTrk(roi, &UtcTrk0bj, continueCount, gSceneMVHdl->m_trajParam.m_histDelta);
				if(!bJudge){
					continueCount = 0;
					_bTrackModel  = SEARCH_MODEL_STATUS;

				}else if(continueCount>_nThredFrms){
					continueCount = 0;
					_bTrackModel  = TRACK_MODEL_STATUS;
					SetReBackFlag(gHistTrkHdl);
				}
			}
		}

		if(_bSimTrack && _bCalSceneMV){
			TRAJECTORY_PARAM	*pTrajParam = &gSceneMVHdl->m_trajParam;
			PointfCR histDelta = gSceneMVHdl->m_trajParam.m_histDelta;
			histDelta.x = -histDelta.x;//direction opposite
			pTrajParam->m_trajAng = _getAngle(histDelta);
		}

#if	(defined SIM_MODEL_1)
		if(_bSimTrack && _bCalSceneMV && utcStatus==1){
			if(_bSimStatus == SEARCH_SIM_STATUS){
				DetectSimilar((void*)gHistTrkHdl, &gSceneMVHdl->m_trajParam, _thrMap,  roi, _tmpl,1);
				if(gHistTrkHdl->bsim){

					UTC_ACQ_param	inputParam;
					inputParam.axisX = UtcTrk0bj.axisX;
					inputParam.axisY = UtcTrk0bj.axisY;
					inputParam.rcWin.x= (int)gHistTrkHdl->simRect.x;
					inputParam.rcWin.y= (int)gHistTrkHdl->simRect.y;
					inputParam.rcWin.width= (int)gHistTrkHdl->simRect.width;
					inputParam.rcWin.height= (int)gHistTrkHdl->simRect.height;
					HistTrkAcq(gHistTrkHdl, *_thrMap, inputParam);

					_bSimStatus = TRACK_SIM_STATUS;
				}
			}else if(_bSimStatus == TRACK_SIM_STATUS){
				HistTrkProc(gHistTrkHdl, *_thrMap, NULL, false);
				if(gHistTrkHdl->utcStatus == 0){//lost
					_bSimStatus = SEARCH_SIM_STATUS;
				}else{//track status
					float diagdist = sqrt(roi.width*roi.width+roi.height*roi.height);
					float dist = sqrt((gHistTrkHdl->_roi.x-roi.x)*(gHistTrkHdl->_roi.x-roi.x)+(gHistTrkHdl->_roi.y-roi.y)*(gHistTrkHdl->_roi.y-roi.y));
					float _overlap = bbOverlap(gHistTrkHdl->_roi, roi);

					if(dist > diagdist){
						_bSimStatus = SEARCH_SIM_STATUS;
					}else if(_overlap>0.80){
						_bSimStatus = OVERLAP_SIM_STATUS;
					}
				}
			}else if(_bSimStatus == OVERLAP_SIM_STATUS){
				;
			}
		}else if(_bSimTrack &&_bCalSceneMV && utcStatus==0){
			if(_bSimStatus != IDLE_SIM_STATUS)
				_bSimStatus = SEARCH_SIM_STATUS;
		}
#elif (defined SIM_MODEL_2)
		if(_bSimTrack &&_bCalSceneMV && utcStatus==1){
			if(_bSimStatus == SEARCH_SIM_STATUS){
				DetectSimilar((void*)(&gSubTrkParam), &gSceneMVHdl->m_trajParam, _thrMap,  roi, _tmpl, 2);
				if(gSubTrkParam.bsim){
					gSubTrkParam._roi = gSubTrkParam.simRect;
					gSubTrkParam.utcStatus = 1;

					_bSimStatus = TRACK_SIM_STATUS;
				}
			}else if(_bSimStatus == TRACK_SIM_STATUS){

				TrackSimilar(&gSubTrkParam, &gSceneMVHdl->m_trajParam, _thrMap,  roi, _tmpl);

				if(gSubTrkParam.utcStatus == 0){//lost
					_bSimStatus = SEARCH_SIM_STATUS;

				}else{//track status
					float _overlap = bbOverlap(gSubTrkParam._roi, roi);
					if(_overlap>0.85){

					}
				}
			}
		}else if(_bSimTrack &&_bCalSceneMV && utcStatus==0){
			if(_bSimStatus != IDLE_SIM_STATUS)
				_bSimStatus = SEARCH_SIM_STATUS;
		}
#elif (defined SIM_MODEL_3)
		if(_bSimTrack &&_bCalSceneMV && utcStatus==1){

			if(_bSimStatus == SEARCH_SIM_STATUS){
				MultiPeakJudge(&gSubTrkParam, &gSceneMVHdl->m_trajParam, _thrMap,  _peakMat, _peakRoi, _peakRes);
				if(gSubTrkParam.bsim){
					gSubTrkParam._roi = gSubTrkParam.simRect;
					gSubTrkParam.utcStatus = 1;

					float _overlap = bbOverlap(gSubTrkParam._roi, roi);
					if(_overlap >0.5)
						_bSimStatus = OVERLAP_SIM_STATUS;
				}
			}else if(_bSimStatus == TRACK_SIM_STATUS){

			}else if(_bSimStatus == OVERLAP_SIM_STATUS){

			}
		}else if(_bSimTrack &&_bCalSceneMV && utcStatus==0){
			if(_bSimStatus != IDLE_SIM_STATUS)
				_bSimStatus = SEARCH_SIM_STATUS;
		}
#endif

		OSA_semSignal(&gBackThrTsk.procWaitSem);
		matFree(_thrMap);
	}

	OSA_printf("%s: exit.", __func__);

	return NULL;
}

#if 0
UTC_RECT_float UtcTrkProc(UTCTRACK_OBJ *pUtcTrkObj, IMG_MAT frame, int *pRtnStat)
{
	UTC_RECT_float rcResult;
	static UInt32 continueCount = 0;

	if(_bPrintTS)
		dbg_tmStat = dbg_getCurTimeInUsec();

	UTILS_assert(frame.width == frame.step[0]);

	backStatus = 1;

	if(_bCalSceneMV){
		if(_bEnhScene){
			getSceneMapEnh(gSceneMVHdl, &frame, 0,  _nrxScene, _nryScene, _fCliplimit);
		}else{
			getSceneMap(gSceneMVHdl, &frame, 0);
		}
		TrackJudge(gSceneMVHdl, 1.0);
	}

	if(_bBlurFilter)
		blurFilterROI(&frame, _roi);
	if(_bEnhROI)
		enhROI(&frame, _roi, 2, 2, _fCliplimit);//local region enhance process

	if(_workMode == _work_mode_serch && _platType == tPLT_WRK ){

		rcResult = UtcTrkSrch(pUtcTrkObj, frame, pRtnStat);

		if(abs(_offsetX)< resTrackObj.res_distance&&  abs(_offsetY)< resTrackObj.res_distance){
			_workMode = _work_mode_trk;
			_bTrkSchLost = false;
		}

		return rcResult;
	}

	if(utcStatus==0 && lostFrame > 1/*3*/)
	{
		if(intervalFrame >0){
			if(_maxpeak > gDynamicParam.retry_acq_thred){
				adaptive_acq_thred = (gDynamicParam.retry_acq_thred < _maxpeak*0.8)?(_maxpeak*0.8):(gDynamicParam.retry_acq_thred);
			}else{
				adaptive_acq_thred = gDynamicParam.retry_acq_thred;
			}
			intervalFrame--;
		}else{
			adaptive_acq_thred = gDynamicParam.retry_acq_thred;
		}
		if(gSerchMode == SEARCH_MODE_ALL){
			rcResult = searchOpenMP(&frame);
		}else if(gSerchMode == SEARCH_MODE_NEAREST){
			rcResult = searchOpenMP_Nearest(&frame);
		}else{
			rcResult = searchOpenMP_LR(&frame, gSerchMode);
		}
		opt_peak_value = opt_peak_value_search;
		if(utcStatus == 1){
			continueCount = 0;
			_bTrackModel  = TRACK_MODEL_STATUS;
			SetReBackFlag(gHistTrkHdl);
		}
	}
	else
	{
		if(0/*_platType == tPLT_WRK /*&& _bsType == BoreSight_Sm*/)
		{
			plat_compensation();
		}

		rcResult = update(&frame);

		if(utcStatus == 0 && lostFrame > 1/*3*/){
			updataROI();
			intervalFrame = gIntervalFrame;
			_bTrackModel = SEARCH_MODEL_STATUS;
			continueCount = 0;
			if(_bBackTrack){
				_roi.x = pUtcTrkObj->axisX - _roi.width/2;
				_roi.y = pUtcTrkObj->axisY - _roi.height/2;
			}
		}

		if(_bBackTrack && _bCalSceneMV && utcStatus==1){//track status process
			if(_bTrackModel == TRACK_MODEL_STATUS){
				backStatus = BackTrackStatus(gHistTrkHdl, gSceneMVHdl, frame, UtcTrk0bj, opt_peak_value, rcResult, utcStatus);
				if(backStatus == 0){
					_bTrackModel = BACK_MODEL_STATUS;
					continueCount = 0;
				}
			}else if(_bTrackModel == BACK_MODEL_STATUS){
				continueCount++;
				bool bJudge = JudgeBackTrk(rcResult, pUtcTrkObj, continueCount, gSceneMVHdl->m_trajParam.m_histDelta);
				if(!bJudge){
					continueCount = 0;
					_bTrackModel  = SEARCH_MODEL_STATUS;
/************************重点关注************************************/
					utcStatus = 0;
					lostFrame = 2;
					updataROI();
					intervalFrame = gIntervalFrame;
					_roi.x = pUtcTrkObj->axisX - _roi.width/2;
					_roi.y = pUtcTrkObj->axisY - _roi.height/2;
/*******************************************************************/
				}else if(continueCount>_nThredFrms){
					continueCount = 0;
					_bTrackModel  = TRACK_MODEL_STATUS;
					SetReBackFlag(gHistTrkHdl);
				}
			}
		}
	}

	_offsetX = (int)(rcResult.x+rcResult.width/2.f) - pUtcTrkObj->axisX;
	_offsetY = pUtcTrkObj->axisY - (int)(rcResult.y+rcResult.height/2.f);

	if(utcStatus == 0)	// 0-lost 1-trk
		*pRtnStat = 2;	// assi
	else
		*pRtnStat = 1;	// trk

	if(_bBackTrack && _bCalSceneMV){//track status process
		if(_bTrackModel == BACK_MODEL_STATUS && utcStatus == 1){
			*pRtnStat = 2;
		}
	}

	if(_bPrintTS){
		frameCount++;
		totalTS += (dbg_getCurTimeInUsec() - dbg_tmStat);
		if(frameCount == 100){
			printf("[DFT Trk] %.3f ms\n", totalTS/1000.f/100.f);
			frameCount = 0;
			totalTS = 0;
		}
	}
	return rcResult;
}
#else
UTC_RECT_float UtcTrkProc(UTCTRACK_OBJ *pUtcTrkObj, IMG_MAT frame, int *pRtnStat)
{
	UTC_RECT_float rcResult;
	static bool 	firstCal = false;
	static bool	firstKalmanAcq = false;

	if(_bPrintTS)
		dbg_tmStat = dbg_getCurTimeInUsec();

	UTILS_assert(frame.width == frame.step[0]);

	if(continueCST>0 && _bKalmanFilter && _bCalSceneMV && _platType == tPLT_WRK){//close loop
		continueCST--;
		TRAJECTORY_PARAM	*pTrajParam = &gSceneMVHdl->m_trajParam;
		rcResult.width =_roi.width;
		rcResult.height = _roi.height;
		rcResult.x = (pUtcTrkObj->axisX-speedCST.x)-rcResult.width/2;
		rcResult.y = (pUtcTrkObj->axisY-speedCST.y)-rcResult.height/2;
		*pRtnStat = 2;
		if(continueCST == 0){
			updataROI2();
			lostFrame = 2;
			intervalFrame = gIntervalFrame*5;
			*pRtnStat = 2;
			utcStatus = 0;
		}
		if(( _bBackTrack ||(_bCalSceneMV && !firstCal)	|| _bSceneMVRecord) && _bStartBackTrk &&_workMode == _work_mode_trk ){
			PointfCR		trkPoint;
			float deltax, deltay;
			TRAJECTORY_PARAM	*pTrajParam = &gSceneMVHdl->m_trajParam;
			memcpy(_tskMap->data_u8, frame.data_u8, _tskMap->size);
			if(_bEnhScene){
				getSceneMapEnh(gSceneMVHdl, _tskMap, 0,  _nrxScene, _nryScene,  _fCliplimit);
			}else{
				getSceneMap(gSceneMVHdl, _tskMap, 0);
			}
			TrackJudge(gSceneMVHdl, 1.0);

			trkPoint.x = pTrajParam->m_trackDelta.x;
			trkPoint.y = pTrajParam->m_trackDelta.y;
			gKalmanTrkObj.KalmanTrkFilter(trkPoint, deltax, deltay);
		}

		return rcResult;
	}else if(continuePeak>0 && _bSimTrack && _bCalSceneMV && _platType == tPLT_WRK){//close loop
		continuePeak--;
		TRAJECTORY_PARAM	*pTrajParam = &gSceneMVHdl->m_trajParam;
		rcResult.width =_roi.width;
		rcResult.height = _roi.height;
		rcResult.x = (pUtcTrkObj->axisX)-rcResult.width/2;
		rcResult.y = (pUtcTrkObj->axisY)-rcResult.height/2;
		*pRtnStat = 2;
		if(continuePeak == 0){
			lostFrame = 0;
			utcStatus = 0;
			*pRtnStat = 2;
			_roi.x = (pTrkObj->axisX-_roi.width/2);
			_roi.y = (pTrkObj->axisY-_roi.height/2);
		}
		return rcResult;
	}

	if(utcStatus==1)
		startFrames++;
	else if(utcStatus==0)
		startFrames = 0;

	if(!_bStartBackTrk && startFrames >18 && _workMode == _work_mode_trk ){
		if(abs(_offsetX)< 30 &&  abs(_offsetY)< 30){
			_bStartBackTrk = true;
			if(_bCalSceneMV	|| _bSceneMVRecord){
				_unInitSceneMV(gSceneMVHdl);
				if(_bEnhScene){
					getSceneMapEnh(gSceneMVHdl, &frame, 1, _nrxScene, _nryScene,  _fCliplimit);
				}else{
					getSceneMap(gSceneMVHdl, &frame, 1);
				}
				clrTrkState(gSceneMVHdl);
				firstCal = true;
				firstKalmanAcq = true;
				_sceneFrmCount = 0;
				_sceneFrmGap = 1;
				_sceneFrmSum = 0;
			}
		}
	}

	if(_bBlurFilter)
		blurFilterROI(&frame, _roi);

	if(( _bBackTrack ||(_bCalSceneMV && !firstCal)	|| _bSceneMVRecord) && _bStartBackTrk &&_workMode == _work_mode_trk ){
		memcpy(_tskMap->data_u8, frame.data_u8, _tskMap->size);
		OSA_semSignal(&gBackThrTsk.procNotifySem);
	}

	if(_bEnhROI)
		enhROI(&frame, _roi, 2, 2, _fCliplimit);//local region enhance process

	if(_workMode == _work_mode_serch && _platType == tPLT_WRK ){
		
		rcResult = UtcTrkSrch(pUtcTrkObj, frame, pRtnStat);

		if(abs(_offsetX)< resTrackObj.res_distance&&  abs(_offsetY)< resTrackObj.res_distance){
			_workMode = _work_mode_trk;
			_bTrkSchLost = false;
		}

		return rcResult;
	}

	if(utcStatus==0 && lostFrame > 1/*3*/)
	{
		if(intervalFrame >0){
			if(_maxpeak > gDynamicParam.retry_acq_thred){
				adaptive_acq_thred = (gDynamicParam.retry_acq_thred < _maxpeak*0.85)?(_maxpeak*0.85):(gDynamicParam.retry_acq_thred);
			}else{
				adaptive_acq_thred = gDynamicParam.retry_acq_thred;
			}
			intervalFrame--;
		}else{
			adaptive_acq_thred = gDynamicParam.retry_acq_thred;
		}
		if(gSerchMode == SEARCH_MODE_ALL){
			rcResult = searchOpenMP(&frame);
		}else if(gSerchMode == SEARCH_MODE_NEAREST){
			rcResult = searchOpenMP_Nearest(&frame);
		}else{
			rcResult = searchOpenMP_LR(&frame, gSerchMode);
		}
		opt_peak_value = opt_peak_value_search;
		_bTrackModel = SEARCH_MODEL_STATUS;
		if(utcStatus == 1){
			continueCount = 0;
			_bTrackModel  = TRACK_MODEL_STATUS;
			SetReBackFlag(gHistTrkHdl);
		}
	}
	else
	{
		if(0/*_platType == tPLT_WRK /*&& _bsType == BoreSight_Sm*/)
		{
			plat_compensation();
		}
		
		rcResult = update(&frame);

		if(utcStatus == 0 && lostFrame > 1/*3*/){
			updataROI();
			intervalFrame = gIntervalFrame;
			continueCount = 0;
			_bTrackModel = SEARCH_MODEL_STATUS;
		}
	}

	_offsetX = (int)(rcResult.x+rcResult.width/2.f) - pUtcTrkObj->axisX;
	_offsetY = pUtcTrkObj->axisY - (int)(rcResult.y+rcResult.height/2.f);

	if(utcStatus == 0)	// 0-lost 1-trk
		*pRtnStat = 2;	// assisearchOpenMP
	else
		*pRtnStat = 1;	// trk

	bool bprintf = true;
	if(_bBackTrack &&  _bStartBackTrk &&  _workMode == _work_mode_trk)
	{
		if(OSA_semWait(&gBackThrTsk.procWaitSem, OSA_TIMEOUT_FOREVER)==OSA_SOK){
			if(_bTrackModel  == SEARCH_MODEL_STATUS && utcStatus==1){
				utcStatus = 0;
				lostFrame = 2;
				updataROI();
				intervalFrame = gIntervalFrame;
				_roi.x = pUtcTrkObj->axisX - _roi.width/2;
				_roi.y = pUtcTrkObj->axisY - _roi.height/2;
			}
			if(_bTrackModel == BACK_MODEL_STATUS && utcStatus == 1){
				*pRtnStat = 2;
			}
			bprintf = false;
		}
	}
	else if(_bKalmanFilter && (_bCalSceneMV && !firstCal) &&  _bStartBackTrk  &&  _workMode == _work_mode_trk)
	{
		if(OSA_semWait(&gBackThrTsk.procWaitSem, OSA_TIMEOUT_FOREVER)==OSA_SOK){
			int	bJudge;
			PointfCR		trkPoint;
			float deltax, deltay, diag;
			TRAJECTORY_PARAM	*pTrajParam = &gSceneMVHdl->m_trajParam;
			trkPoint.x = pTrajParam->m_trackDelta.x;
			trkPoint.y = pTrajParam->m_trackDelta.y;
			if(firstKalmanAcq){

				gKalmanTrkObj.KalmanTrkAcq(trkPoint, _KalmanCoefQ, _KalmanCoefR);
				firstKalmanAcq = false;
			}else{
				PointfCR		mvThred, stillThred;
				float slopeThred;
				if(frame.width <= 768){
					mvThred.x = _KalmanMVThred.x/2;			mvThred.y = _KalmanMVThred.y/2;
					stillThred.x = _KalmanStillThred.x/2;		stillThred.y = _KalmanStillThred.y/2;
					slopeThred = _slopeThred*0.6;
				}else if(frame.width <= 1280){
					mvThred.x = _KalmanMVThred.x;			mvThred.y = _KalmanMVThred.y;
					stillThred.x = _KalmanStillThred.x;		stillThred.y = _KalmanStillThred.y;
					slopeThred = _slopeThred;
				}else{
					mvThred.x = _KalmanMVThred.x*1.5;			mvThred.y = _KalmanMVThred.y;
					stillThred.x = _KalmanStillThred.x*1.5;		stillThred.y = _KalmanStillThred.y;
					slopeThred = _slopeThred*1.5;
				}

				gKalmanTrkObj.KalmanTrkFilter(trkPoint, deltax, deltay);
				bJudge = gKalmanTrkObj.KalmanTrkJudge(mvThred, stillThred, slopeThred, &speedCST, _bSceneMVRecord);

				if(bJudge && utcStatus==1){
					continueCST = 40;
					_sceneFrmCount = 0;
					_sceneFrmGap = 1;
					_sceneFrmSum = 0;
					 gKalmanTrkObj.KalmanTrkReset();
					 if(_bSceneMVRecord){
						 printf("%s:kalman predict is abnormal! bJudge=%d\n",__func__, bJudge);
					 }
				}
			}
			if(_bSimTrack && _bSimStatus == OVERLAP_SIM_STATUS && utcStatus==1&& continueCST==0 ){
				continuePeak = 40;
				_sceneFrmCount = 0;
				_sceneFrmGap = 1;
				_sceneFrmSum = 0;
				_bSimStatus = SEARCH_SIM_STATUS;
				 if(_bSceneMVRecord){
					 printf("%s: multi peak is abnormal!\n",__func__);
				 }
			}
		}
	}
	else if(_bSimTrack &&_bCalSceneMV &&  _workMode == _work_mode_trk)
	{
		if(_bSimTrack && _bSimStatus == OVERLAP_SIM_STATUS && utcStatus==1){
			continuePeak = 40;
			_sceneFrmCount = 0;
			_sceneFrmGap = 1;
			_sceneFrmSum = 0;
			_bSimStatus = SEARCH_SIM_STATUS;
			 if(_bSceneMVRecord){
				 printf("%s: multi peak is abnormal!\n",__func__);
			 }
		}
	}
	else if(_bSceneMVRecord &&  _bStartBackTrk &&  _workMode == _work_mode_trk)
	{
		if(OSA_semWait(&gBackThrTsk.procWaitSem, OSA_TIMEOUT_FOREVER)==OSA_SOK){
			;
		}
	}
	firstCal  = false;

	if(bprintf){
		;//printf("%s:no wait signal! \n",__func__);
	}

	if(_bPrintTS){
		frameCount++;
		totalTS += (dbg_getCurTimeInUsec() - dbg_tmStat);
		if(frameCount == 100){
			printf("[DFT Trk] %.3f ms\n", totalTS/1000.f/100.f);
			frameCount = 0;
			totalTS = 0;
		}
	}
	return rcResult;
}
#endif


static bool DetectAanalyze(UTC_Rect acqWin, UTC_Rect  srRect, Pattern  *pPatterns, int	Num, UTC_Rect *pAcqRect)
{
	int i, szIdx = 0, distIdx = 0;
	Pattern	maxSizePat, minDistPat, acqPat;
	int sz, dist, maxSZ = 0, minDist = srRect.width*srRect.height, minDistSZ, maxDist;
	PointICR		acqCent, patCent;
	UTC_SIZE	patSize;
	assert(gConRegHdl != NULL && pPatterns != NULL);

	acqCent.x = acqWin.x + acqWin.width/2;
	acqCent.y = acqWin.y + acqWin.height/2;

	if(Num == 0){
		*pAcqRect = acqWin;
		return false;
	}else if(Num == 1){
		if(pAcqRect != NULL){
			patCent.x = (pPatterns[0].rightbottom.x + pPatterns[0].lefttop.x)/2;
			patCent.y = (pPatterns[0].rightbottom.y + pPatterns[0].lefttop.y)/2;
			patSize.width = pPatterns[0].rightbottom.x - pPatterns[0].lefttop.x;
			patSize.height = pPatterns[0].rightbottom.y - pPatterns[0].lefttop.y;
#if 0
			pAcqRect->width =(abs(patCent.x-acqCent.x))*2+patSize.width;
			pAcqRect->height = (abs(patCent.y-acqCent.y))*2+patSize.height;
			pAcqRect->x = acqCent.x - pAcqRect->width/2;
			pAcqRect->y = acqCent.y - pAcqRect->height/2;
#else
			pAcqRect->width = patSize.width;
			pAcqRect->height = patSize.height;
			pAcqRect->x	 = patCent.x - pAcqRect->width/2;
			pAcqRect->y	 = patCent.y - pAcqRect->height/2;
#endif
		}
		return true;
	}

	for(i=0; i<Num; i++){
		patCent.x = (pPatterns[i].rightbottom.x+pPatterns[i].lefttop.x)/2;
		patCent.y = (pPatterns[i].rightbottom.y+pPatterns[i].lefttop.y)/2;
		sz = (pPatterns[i].rightbottom.x-pPatterns[i].lefttop.x)*(pPatterns[i].rightbottom.x-pPatterns[i].lefttop.x);
		dist = (acqCent.x-patCent.x)*(acqCent.x-patCent.x)+(acqCent.y-patCent.y)*(acqCent.y-patCent.y);
		if(sz>maxSZ){	maxSZ = sz;	szIdx = i; maxDist = dist;}
		if(dist<minDist){minDist = dist;	distIdx = i;	minDistSZ = sz;}
	}
	maxSizePat = pPatterns[szIdx];
	minDistPat = pPatterns[distIdx];
	if(maxSZ > minDistSZ*4){
		acqPat = maxSizePat;
	}else{
		acqPat = minDistPat;
	}
//	acqPat = minDistPat;
	if(pAcqRect != NULL){
		patCent.x = (acqPat.rightbottom.x + acqPat.lefttop.x)/2;
		patCent.y = (acqPat.rightbottom.y + acqPat.lefttop.y)/2;
		patSize.width = acqPat.rightbottom.x - acqPat.lefttop.x;
		patSize.height = acqPat.rightbottom.y - acqPat.lefttop.y;
#if 0
		pAcqRect->width =(abs(patCent.x-acqCent.x))*2+patSize.width;
		pAcqRect->height = (abs(patCent.y-acqCent.y))*2+patSize.height;
		pAcqRect->x = acqCent.x - pAcqRect->width/2;
		pAcqRect->y = acqCent.y - pAcqRect->height/2;
#else
		pAcqRect->width = patSize.width;
		pAcqRect->height = patSize.height;
		pAcqRect->x	 = patCent.x - pAcqRect->width/2;
		pAcqRect->y	 = patCent.y - pAcqRect->height/2;
#endif
///////////////////////////////////////////////////////////////////////////////////////
	}
	return true;
}

#define		SET_P_RECT(pRect)	\
	if(pRect != NULL){		\
			pRect->x = inputParam.rcWin.x;	\
			pRect->y = inputParam.rcWin.y;	\
			pRect->width = inputParam.rcWin.width;	\
			pRect->height = inputParam.rcWin.height;		\
		}

static bool	bAcqTarget = true;
static bool	bMVDetect = false;
static UInt32		nMVFrames = 0;
bool	UtcAcqTarget(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param inputParam, UTC_Rect *pRect)
{
	assert(gSceneMVHdl != NULL);
	Pattern  		pPatterns[SAMPLE_NUMBER];
	int				i,	patternNum;
	UTC_Rect 	acqRect;

	if(bAcqTarget){
		_unInitSceneMV(gSceneMVHdl);
		if(_bEnhScene){
			getSceneMapEnh(gSceneMVHdl, &frame, 1, _nrxScene, _nryScene, _fCliplimit);
		}else{
			getSceneMap(gSceneMVHdl, &frame, 1);
		}
		bAcqTarget = false;
		SET_P_RECT(pRect);
		_sceneFrmCount = 0;
		_sceneFrmGap = 1;
		_sceneFrmSum = 0;
		return false;
	}else{
		if(_bEnhScene){
			getSceneMapEnh(gSceneMVHdl, &frame, 0,  _nrxScene, _nryScene,  _fCliplimit);
		}else{
			getSceneMap(gSceneMVHdl, &frame, 0);
		}
		TrackJudge(gSceneMVHdl, 1.0);
	}

	PointfCR	 hist_delta;
	TRAJECTORY_PARAM	*pTrajParam = &gSceneMVHdl->m_trajParam;
	int	histNum = (_detectorHistFrms<pTrajParam->m_moveJudge.history_num)?_detectorHistFrms:pTrajParam->m_moveJudge.history_num;
	if(pTrajParam->m_histNum<histNum){
		SET_P_RECT(pRect);
		return false;
	}

	hist_delta.x = hist_delta.y = 0.f;
	for(i=0; i<histNum; i++){
		hist_delta.x += pTrajParam->m_historyPos[pTrajParam->m_histNum-i-1].x;
		hist_delta.y += pTrajParam->m_historyPos[pTrajParam->m_histNum-i-1].y;
	}
	hist_delta.x = hist_delta.x/histNum;
	hist_delta.y = hist_delta.y/histNum;

	if(fabs(hist_delta.x)>_stillThred || fabs(hist_delta.y)>_stillThred){
		SET_P_RECT(pRect);
		bMVDetect = true;
		nMVFrames = 0;
		gMVDetectObj.setNFrames(0);
		return false;
	}

	cv::Mat	midFrame, detectFrame;
	cv::Rect	detectROI, imgRect, roiRect;
	IMG_MAT	detectMap;
	UTC_Rect	detectRect;

	midFrame = cv::Mat(frame.height, frame.width, CV_8UC1, frame.data_u8);
	detectROI.width = inputParam.rcWin.width*1.2;
	detectROI.height = inputParam.rcWin.height*1.2;
	detectROI.x = (inputParam.rcWin.x + inputParam.rcWin.width/2) - detectROI.width/2;
	detectROI.y = (inputParam.rcWin.y + inputParam.rcWin.height/2) - detectROI.height/2;
	imgRect.x = imgRect. y= 0;
	imgRect.width = frame.width;
	imgRect.height = frame.height;
	_overlapRoi(detectROI, imgRect, &roiRect);
	detectROI = roiRect;

	midFrame(detectROI).copyTo(detectFrame);
	gMVDetectObj.DetectProcess(detectFrame);

	nMVFrames++;
	if(nMVFrames<_stillFrms){
		SET_P_RECT(pRect);
		return false;
	}else if(bMVDetect){
		bMVDetect = false;
		SET_P_RECT(pRect);
		return false;
	}

	detectMap.channels = 1;	detectMap.dtype = 0;	detectMap.data_u8 = gMVDetectObj.fgmask[0].data;
	detectMap.width = gMVDetectObj.fgmask[0].cols; detectMap.height = gMVDetectObj.fgmask[0].rows;
	detectMap.step[0] = gMVDetectObj.fgmask[0].step;
	detectMap.size = detectMap.width*detectMap.height;
	assert(detectMap.width == detectMap.step[0]);

	detectRect.x = inputParam.rcWin.x-detectROI.x;
	detectRect.y = inputParam.rcWin.y-detectROI.y;
	detectRect.width = inputParam.rcWin.width;
	detectRect.height = inputParam.rcWin.height;

	int	iRtn = GetMoveDetect(gConRegHdl, detectMap, detectRect, _salientThred, _salientScatter);
	if(iRtn){
		for(i=0; i<gConRegHdl->m_patternnum; i++){
			pPatterns[i].lefttop.x = gConRegHdl->m_pPatterns[i].lefttop.x + detectROI.x;
			pPatterns[i].lefttop.y = gConRegHdl->m_pPatterns[i].lefttop.y + detectROI.y;
			pPatterns[i].rightbottom.x = gConRegHdl->m_pPatterns[i].rightbottom.x + detectROI.x;
			pPatterns[i].rightbottom.y = gConRegHdl->m_pPatterns[i].rightbottom.y + detectROI.y;
		}
		patternNum = gConRegHdl->m_patternnum;
		detectRect.x = detectROI.x;								detectRect.y = detectROI.y;
		detectRect.width = inputParam.rcWin.width;	detectRect.height = inputParam.rcWin.height;

		iRtn = DetectAanalyze(inputParam.rcWin, detectRect, pPatterns, patternNum, &acqRect);
		if(pRect != NULL)
			*pRect = acqRect;
	}

	return iRtn;
}

void UtcGetOptValue(UTCTRACK_OBJ *pUtcTrkObj,  float *optValue)
{
	if(optValue != NULL)
		*optValue = opt_peak_value;
}

void UtcSetDynParam(UTCTRACK_OBJ *pUtcTrkObj, UTC_DYN_PARAM dynamicParam)
{
	memcpy(&gDynamicParam , &dynamicParam, sizeof(UTC_DYN_PARAM));
	memcpy(&gHistTrkHdl->gDynamicParam , &dynamicParam, sizeof(UTC_DYN_PARAM));
	printf("%s:Dynamic Param--occlusion_thred:%f, retry_acq_thred:%f \n",__func__, gDynamicParam.occlusion_thred, gDynamicParam.retry_acq_thred);
}

void UtcSetUpFactor(UTCTRACK_OBJ *pUtcTrkObj, float up_factor)
{
	interp_factor = up_factor;
	printf("%s: interp_factor=%f \n",__func__, interp_factor);
}

void UtcSetSerchMode(UTCTRACK_OBJ *pUtcTrkObj, SEARCH_MODE_TYPE searchMode)
{
	if(searchMode >= SEARCH_MODE_ALL && searchMode <SEARCH_MODE_MAX){
		gSerchMode = searchMode;
		printf("%s:Track SearchMode:%s\n",__func__, strMode[gSerchMode]);
	}else{
		printf("%s:Do not Set SearchMode \n",__func__);
	}
}

void UtcSetIntervalFrame(UTCTRACK_OBJ *pUtcTrkObj, int gapFrame)
{
	gIntervalFrame = gapFrame;
	printf("%s:Set Interval Frame:%d \n",__func__, gIntervalFrame);
}

void UtcSetSearchParam(UTCTRACK_OBJ *pUtcTrkObj, UTC_SEARCH_PARAM searchParam)
{
	min_bias_pix	= searchParam.min_bias_pix;
	max_bias_pix	 = searchParam.max_bias_pix;
	_blendValue	=  searchParam.blend_value;
}

void UtcSetPLT_BS(UTCTRACK_OBJ *pUtcTrkObj, tPLT pltWork, BS_Type bsType)
{
	_platType = pltWork;
	_bsType = bsType;
	printf("%s:Set _platType:%d--_bsType:%d\n",__func__, _platType, _bsType);
}

void UtcSetRestraint(UTCTRACK_OBJ *pUtcTrkObj,TRK_SECH_RESTRAINT resTraint)
{
	memcpy(&resTrackObj, &resTraint, sizeof(TRK_SECH_RESTRAINT));
	printf("%s:Set _distance:%d--_area:%d\n",__func__, resTrackObj.res_distance, resTrackObj.res_area);
}

void UtcSetEnhance(UTCTRACK_OBJ *pUtcTrkObj, bool	bEnable)
{
	_bEnhROI = bEnable;
	printf("%s:Set Enhance enable:%d \n",__func__, _bEnhROI);
}

void UtcSetBlurFilter(UTCTRACK_OBJ *pUtcTrkObj, bool	bEnable)
{
	_bBlurFilter = bEnable;
	printf("%s:Set Blur Filter enable:%d \n",__func__, _bBlurFilter);
}

void UtcSetEnhfClip(UTCTRACK_OBJ *pUtcTrkObj, float fCliplimit)
{
	_fCliplimit = fCliplimit;
	printf("%s:Set Enhance fClip:%f \n",__func__, _fCliplimit);
}

void UtcSetPrintTS(UTCTRACK_OBJ *pUtcTrkObj, bool bPrint)
{
	_bPrintTS = bPrint;
	printf("%s:Set Enable print time:%d \n",__func__, _bPrintTS);
}

void UtcSetPredict(UTCTRACK_OBJ *pUtcTrkObj, bool bPredict)
{
	_bPredict= bPredict;
	printf("%s:Set Enable move predict:%d \n",__func__, _bPredict);
}

void UtcSetMvPixel(UTCTRACK_OBJ *pUtcTrkObj,  int mvPixelX, int mvPixelY)
{
	_iMvPixel.x = mvPixelX;
	_iMvPixel.y = mvPixelY;
	printf("%s:Set move pixel(x,y):(%d,%d) \n",__func__, _iMvPixel.x, _iMvPixel.y);
}

void UtcSetMvPixel2(UTCTRACK_OBJ *pUtcTrkObj,  int mvPixelX2, int mvPixelY2)
{
	_iMvPixel2.x = mvPixelX2;
	_iMvPixel2.y = mvPixelY2;
	printf("%s:Set move pixel2(x,y):(%d,%d) \n",__func__, _iMvPixel2.x, _iMvPixel2.y);
}

void UtcSetSegPixelThred(UTCTRACK_OBJ *pUtcTrkObj,  int segPixelX, int segPixelY)
{
	_iSegPixel.x = segPixelX;
	_iSegPixel.y = segPixelY;
	printf("%s:Set seg pixel thred(x,y):(%d,%d) \n",__func__, _iSegPixel.x, _iSegPixel.y);
}

void UtcSetRoiMaxWidth(UTCTRACK_OBJ *pUtcTrkObj, int maxValue)
{
	_roiMaxWidth = maxValue;
	if(_roiMaxWidth < 16 || _roiMaxWidth > 0xFFFFFFF)
		_roiMaxWidth = 0xFFFFFFF;
	printf("%s: Set Roi max width = %d  to %d\n",__func__, maxValue, _roiMaxWidth);
}

void UtcSetSalientThred(UTCTRACK_OBJ *pUtcTrkObj, int  thred)
{
	_salientThred = clip(10, 180, thred);

	printf("%s: Set salient gray thred=%d clip thred=%d\n",__func__, thred, _salientThred);
}

void UtcSetSalientScatter(UTCTRACK_OBJ *pUtcTrkObj, int scatter)
{
	_salientScatter = clip(10, 200, scatter);

	printf("%s: Set salient scatter = %d clip scatter = %d\n",__func__, scatter, _salientScatter);
}

void UtcSetSalientScaler(UTCTRACK_OBJ *pUtcTrkObj,  int large, int mid, int small)
{
	assert(large >= mid && mid >= small);
	_smallScaler = clip(32, 192, small);
	_midScaler = clip(96, 384, mid);
	_largeScaler = clip(128, 512, large);
	printf("%s: Set salient large scaler = %d mid scaler = %d small scaler = %d\n",__func__, _largeScaler,_midScaler,  _smallScaler);
}

void UtcSetMultScaler(UTCTRACK_OBJ *pUtcTrkObj, int bMultScaler)
{
	_bmultScaler = (bMultScaler==0)?0:1;
	printf("%s: Set salient mult or single scaler= %d\n",__func__, _bmultScaler);
}

void UtcSetSRMinAcqSize(UTCTRACK_OBJ *pUtcTrkObj, UTC_SIZE minAcqSize)
{
	_minAcqSize = minAcqSize;
	printf("%s: Set salient min acq size w=%d,h=%d\n",__func__, _minAcqSize.width, _minAcqSize.height);
}

void UtcSetSRAcqRatio(UTCTRACK_OBJ *pUtcTrkObj, float  ratio)
{
	_acqRatio = ratio;
	printf("%s: Set salient acq ratio=%0.2f\n",__func__, _acqRatio);
}
void UtcSetBigSearch(UTCTRACK_OBJ *pUtcTrkObj, bool  bBigSearch)
{
	_bBigSearch = bBigSearch;
	printf("%s: Set Enable Big Search=%d\n",__func__, _bBigSearch);
}
void UtcSetDynamicRatio(UTCTRACK_OBJ *pUtcTrkObj, bool  bDynamic)
{
	_bDynamicRatio = bDynamic;
	printf("%s: Set Enable Dynamic Ratio=%d\n",__func__, _bDynamicRatio);
}
void UtcSetSceneMV(UTCTRACK_OBJ *pUtcTrkObj, bool  bSceneMV)
{
	_bCalSceneMV = bSceneMV;
	printf("%s: Set Enable Cal Scene Move=%d\n",__func__, _bCalSceneMV);
}
void UtcSetBackTrack(UTCTRACK_OBJ *pUtcTrkObj, bool  bBackTrack)
{
	_bBackTrack = bBackTrack;
	printf("%s: Set Enable Back Track=%d\n",__func__, _bBackTrack);
}
void UtcGetSceneMV(UTCTRACK_OBJ *pUtcTrkObj, float *speedx, float *speedy)
{
#if 1
	if(speedx != NULL)
		*speedx = gSceneMVHdl->m_trajParam.m_trackDelta.x;
	if(speedy != NULL)
		*speedy = gSceneMVHdl->m_trajParam.m_trackDelta.y;
#else
	TRAJECTORY_PARAM	*pTrajParam = &gSceneMVHdl->m_trajParam;
	if(pTrajParam->m_histNum>0){
		if(speedx != NULL)
			*speedx = pTrajParam->m_historyPos[pTrajParam->m_histNum-1].x;
		if(speedy != NULL)
			*speedy = pTrajParam->m_historyPos[pTrajParam->m_histNum-1].y;
	}else{
		if(speedx != NULL)
			*speedx = gSceneMVHdl->m_trajParam.m_histDelta.x;
		if(speedy != NULL)
			*speedy = gSceneMVHdl->m_trajParam.m_histDelta.y;
	}
#endif
}
void UtcSetBackThred(UTCTRACK_OBJ *pUtcTrkObj, int  nThredFrms, float fRatioFrms)
{
	_nThredFrms = nThredFrms;
	_fRatioFrms = fRatioFrms;
	printf("%s: Set Back ThredFrms=%d, fRatioFrms=%0.2f\n",__func__, _nThredFrms, _fRatioFrms);
}
void UtcSetAveTrkPos(UTCTRACK_OBJ *pUtcTrkObj, bool  bAveTrkPos)
{
	_bAveTrkPos = bAveTrkPos;
	printf("%s: Set Enable Ave Trk Pos Caculate=%d\n",__func__, _bAveTrkPos);
}
void UtcSetDetectftau(UTCTRACK_OBJ *pUtcTrkObj, float  fTau)
{
	_detectfTau = fTau;
	gMVDetectObj.setShadowThreshold(_detectfTau);
	printf("%s: Set  shadow threshold=%0.2f\n",__func__, _detectfTau);
}
void UtcSetDetectBuildFrms(UTCTRACK_OBJ *pUtcTrkObj, int  buildFrms)
{
	_detectBuildFrms = buildFrms;
	gMVDetectObj.setHistory(_detectBuildFrms);
	printf("%s: Set  Building Frames=%d\n",__func__, _detectBuildFrms);
}

void UtcSetLostFrmThred(UTCTRACK_OBJ *pUtcTrkObj, int  LostFrmThred)
{
	_LostFrmThred = LostFrmThred;
	printf("%s: Set  lost frames threshold=%d\n",__func__, _LostFrmThred);
}
void UtcSetHistMVThred(UTCTRACK_OBJ *pUtcTrkObj, float  histMvThred)
{
	_histMvThred = histMvThred;
	printf("%s: Set  hist scene move threshold=%0.2f\n",__func__, _histMvThred);
}
void UtcSetDetectFrmsThred(UTCTRACK_OBJ *pUtcTrkObj, int  detectFrms)
{
	_detectorHistFrms = detectFrms;
	printf("%s: Set  move detect Frames threshold=%d\n",__func__, _detectorHistFrms);
}
void UtcSetStillFrmsThred(UTCTRACK_OBJ *pUtcTrkObj, int  stillFrms)
{
	_stillFrms = stillFrms;
	printf("%s: Set  still wait Frames threshold=%d\n",__func__, _stillFrms);
}
void UtcSetStillPixThred(UTCTRACK_OBJ *pUtcTrkObj, float  stillThred)
{
	_stillThred = stillThred;
	printf("%s: Set  still pixel threshold=%0.2f\n",__func__, _stillThred);
}
void UtcSetKalmanFilter(UTCTRACK_OBJ *pUtcTrkObj, bool  bKalmanFilter)
{
	_bKalmanFilter = bKalmanFilter;
	printf("%s: Set Enable Kalman Filter=%d\n",__func__, _bKalmanFilter);
}
void UtcSetKFMVThred(UTCTRACK_OBJ *pUtcTrkObj, float xMVThred, float yMVThred)
{
	_KalmanMVThred.x = xMVThred;
	_KalmanMVThred.y = yMVThred;
	printf("%s: Set Kalman Filter MVThred x=%0.2f,y=%0.2f\n",__func__, _KalmanMVThred.x, _KalmanMVThred.y);
}
void UtcSetKFStillThred(UTCTRACK_OBJ *pUtcTrkObj, float xMVThred, float yMVThred)
{
	_KalmanStillThred.x = xMVThred;
	_KalmanStillThred.y = yMVThred;
	printf("%s: Set Kalman Filter Still Thred x=%0.2f,y=%0.2f\n",__func__, _KalmanStillThred.x, _KalmanStillThred.y);
}
void UtcSetKFSlopeThred(UTCTRACK_OBJ *pUtcTrkObj, float slopeThred)
{
	_slopeThred = slopeThred;
	printf("%s: Set Kalman Filter Slope Thred = %0.2f\n",__func__, _slopeThred);
}
void UtcSetKFHistThred(UTCTRACK_OBJ *pUtcTrkObj, float kalmanHistThred)
{
	_KalmanHistThred = kalmanHistThred;
	printf("%s: Set Kalman Filter HistThred=%0.2f\n",__func__, _KalmanHistThred);
}
void UtcSetKFCoefQR(UTCTRACK_OBJ *pUtcTrkObj, float kalmanCoefQ, float kalmanCoefR)
{
	_KalmanCoefQ = kalmanCoefQ;
	_KalmanCoefR = kalmanCoefR;
	printf("%s: Set Kalman Filter Coef Q=%f, R=%f\n",__func__, _KalmanCoefQ, _KalmanCoefR);
}
void UtcSetSceneMVRecord(UTCTRACK_OBJ *pUtcTrkObj, bool  bSceneMVRecord)
{
	_bSceneMVRecord = bSceneMVRecord;
	printf("%s: Set Enable Scene MV Record=%d\n",__func__, _bSceneMVRecord);
}
void UtcSetTrkAngleThred(UTCTRACK_OBJ *pUtcTrkObj, float trkAngleThred)
{
	_trkAngleThred = clip(30, 180, trkAngleThred);
	printf("%s: Set Trk Anlge Thred=%0.2f\n",__func__, _trkAngleThred);
}
void UtcSetTrkHighOpt(UTCTRACK_OBJ *pUtcTrkObj, bool bHighOpt)
{
	_bHighOpt = bHighOpt;
	printf("%s: Set Enable Trk Hight Opt=%d\n",__func__, _bHighOpt);
}
void UtcSetSceneEnh(UTCTRACK_OBJ *pUtcTrkObj, bool bEnhScene)
{
	_bEnhScene = bEnhScene;
	printf("%s: Set Enable Scene MV ENHANCE=%d\n",__func__, _bEnhScene);
}
void UtcSetSceneEnhMacro(UTCTRACK_OBJ *pUtcTrkObj, unsigned int	 nrxScene, unsigned int nryScene)
{
	_nrxScene = nrxScene;
	_nryScene = nryScene;
	printf("%s: Set Enable Scene ENHANCE MACRO nix=%d, niy=%d,\n",__func__, _nrxScene, _nryScene);
}
void UtcNotifyServoSpeed(UTCTRACK_OBJ *pUtcTrkObj, float speedx, float speedy)
{
	_servoSpeed.x = speedx;
	_servoSpeed.y = speedy;
}
void UtcSetPeakMVThred(UTCTRACK_OBJ *pUtcTrkObj, float  peakMvThred)
{
	_peakMvThred = peakMvThred;
	printf("%s: Set  hist mulit peak move threshold=%0.2f\n",__func__, _peakMvThred);
}
