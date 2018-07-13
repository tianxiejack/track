#ifndef 		_HIST_TRACK_HEAD_
#define		_HIST_TRACK_HEAD_

#include "PCTracker.h"
#include "UtcTrack.h"
#include "BackTrk.h"
#include "SceneMV.h"

#define 	HIST_SAVE_NUM			(30)

typedef struct _hist_trk_obj{

	UTCTRACK_OBJ	UtcTrkObj;

	UTC_SIZE _tmpl_sz;
	UTC_SIZE _match_sz;
	float	_scale, _scaleBK,_maxpeak;
	int		size_patch[3];

	IMG_MAT *_FeaturesMap;
	IMG_MAT *_tmpl;
	IMG_MAT *_hann;
	IMG_MAT *_alphaf;
	IMG_MAT *_prob;
	IMG_MAT *_feature_tmpl[HIST_SAVE_NUM];
	IMG_MAT *_feature_alphaf[HIST_SAVE_NUM];
	IMG_MAT	*_templateMap;
	IMG_MAT	*_curImageMap;
	float _feature_peak[HIST_SAVE_NUM];

	float opt_peak_value;// = 0.0f;
	float opt_peak_value_search;// = 0.0;
	float opt_apceVal_search;//0.0;
	UTC_RECT_float opt_roi_search;

	UTC_RECT_float _roi, _roiBK;

	float	mult_peak_value;
	float	mult_peak_apce;
	UTC_RECT_float		mult_peak_Wnd;

	int	 featIndx;// = 0;
	int   lostFrame;// = 0;
	int   utcStatus;// = 0;

	DFT_Trk_Param			dftTrkParam;
	UTC_DYN_PARAM		gDynamicParam;

	int		intervalFrame;
	int		gIntevlFrame;
	float	adaptive_acq_thred;// = RETRY_ACQ_THRED;

	int		_offsetX, _offsetY;

	BACK_HANDLE	m_backHdl;

	bool	bsim;//is have similar target in the trajectory
	UTC_RECT_float		simRect;

}HIST_TRK_Obj ,*HIST_TRACK_HANDLE;


HIST_TRACK_HANDLE CreateHistTrk(bool multiscale, bool fixed_window);

void DestroySceneTrk(HIST_TRACK_HANDLE handle);

UTC_RECT_float HistTrkAcq(HIST_TRACK_HANDLE pHistTrkObj, IMG_MAT frame, UTC_ACQ_param inputParam);

UTC_RECT_float HistTrkProc(HIST_TRACK_HANDLE pHistTrkObj, IMG_MAT frame, int *pRtnStat, bool bUpdata);

void SetReBackFlag(HIST_TRACK_HANDLE pHistTrkObj);

void BackFrmRecord(HIST_TRACK_HANDLE pHistTrkObj, IMG_MAT frame, float opt, UTC_RECT_float utcRect, int utcStatus, PointfCR sceneVelocity);

int	 BackTrackStatus(	HIST_TRACK_HANDLE pHistTrkObj, SCENE_MV_HANDLE pSceneMVObj, IMG_MAT frame, UTCTRACK_OBJ UtcTrkObj,
											float opt, UTC_RECT_float roi, int utcStatus);

UTC_RECT_float BackTrkProc(HIST_TRACK_HANDLE pHistTrkObj, IMG_MAT frame, float opt, UTC_RECT_float utcRect, int utcStatus, PointfCR histVelocity, int *pRtnStat);

#endif
