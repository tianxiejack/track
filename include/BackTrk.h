#ifndef		_BACK_TRK_HEAD_
#define		_BACK_TRK_HEAD_

#include "PCTracker.h"
#include "UtcTrack.h"
#define		BACK_FRAMES_NUM	(24)

typedef enum{
	BACK_TRK_TYPE	=	0x01,
	BACK_SECH_TYPE,
}BACK_ENUM_TYPE;

typedef struct _back_obj_{

	IMG_MAT	*_histFrames[BACK_FRAMES_NUM];
	UTC_RECT_float		_histRectPos[BACK_FRAMES_NUM];//DFT Trk pos
	float 	_histOpt[BACK_FRAMES_NUM];//DFT Trk result
	int	  	_histStatus[BACK_FRAMES_NUM];
	PointfCR		_histVelocity;
	int	  	nFrames;
	int		nFrmIndx;
	int 		nWidth, nHeight;

	float	trkPosOpt[BACK_FRAMES_NUM];
	UTC_RECT_float		trkPos[BACK_FRAMES_NUM];//cur pos
	PointfCR					sceneVelocity[BACK_FRAMES_NUM];
	IMG_MAT 				*_tmplateMap;
	IMG_MAT				*_curImgMap;
	UTC_RECT_float		_roi;
	UTC_RECT_float		_curroi;
	int   						backStatus;// = 0;

	UTC_SIZE				 _addSize;
	int							_gapFrames;
	bool						_reAcqFlag;
	bool						_bConf;

}BACK_OBJ, *BACK_HANDLE;

BACK_HANDLE	CreatBackHandle();

void	DestroyBackHandle(BACK_HANDLE backHdl);

void initBackObj(BACK_OBJ *pBackObj, UTC_RECT_float roi, IMG_MAT *image);


#endif
