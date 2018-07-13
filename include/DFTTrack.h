#ifndef		_DFT_TRACK_HEAD_
#define		_DFT_TRACK_HEAD_

#include "DFT.h"

#define  FEATURE_BK_NUM			15

typedef struct _dft_param_t{

	float interp_factor; // linear interpolation factor for adaptation
	float sigma; // gaussian kernel bandwidth
	float lambda; // regularization
	int cell_size; // HOG cell size
	//static int cell_sizeQ; // cell size^2, to avoid repeated operations
	float padding; // extra area surrounding the target
	float output_sigma_factor; // bandwidth of gaussian target
	int template_size; // template size
	float scale_step; // scale step for multi-scale estimation
	float scale_weight;  // to downweight detection scores of other scales for added stability

}DFT_Trk_Param;

typedef struct _dft_trk_obj{
	int		bInited;

	IMG_MAT_FLOAT	_alphaf;
	IMG_MAT_FLOAT	_prob;
	IMG_MAT_FLOAT	_tmpl;

	IMG_MAT_FLOAT	_feature_alphaf[FEATURE_BK_NUM];
	IMG_MAT_FLOAT	_feature_tmpl[FEATURE_BK_NUM];
	float			_feature_peak[FEATURE_BK_NUM];
	int				featIndx;

	float		opt_peak_value;
	float		opt_peak_value_search;

	float		interp_factor; // linear interpolation factor for adaptation
	float		sigma; // gaussian kernel bandwidth
	float		lambda; // regularization
	int			size_patch[3];
	bool			_hogfeatures;

	float		padding; // extra area surrounding the target
	float		output_sigma_factor; // bandwidth of gaussian target

}DFT_TRK_Obj ,*DFT_TRACK_HANDLE;

DFT_TRACK_HANDLE		DFT_TRK_Create();
void DFT_TRK_init(DFT_TRACK_HANDLE	trkHandle,	IMG_MAT_FLOAT	_alphaf,	IMG_MAT_FLOAT _prob, IMG_MAT_FLOAT _tmpl);
void DFT_TRK_unInit(DFT_TRACK_HANDLE	trkHandle);
void DFT_TRK_setParam(DFT_TRACK_HANDLE	trkHandle, DFT_Trk_Param _prama);
void DFT_TRK_detect(DFT_TRACK_HANDLE	trkHandle, IMG_MAT_FLOAT feature, float *peak_value, PointfCR *ptPos);
void DFT_TRK_train(DFT_TRACK_HANDLE	trkHandle, IMG_MAT_FLOAT feature, float train_interp_factor);

#endif
