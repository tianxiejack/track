#ifndef		_SCENE_MV_HEAD_
#define		_SCENE_MV_HEAD_

#include "PCTracker.h"
#include "DFT.h"
#include "DFTTrack.h"
#include "HogFeat.h"
#include "RectMat.h"
#include "malloc_align.h"

#define		HISTORY_POS_NUM			80//max history frames cannot reset
#define		TRACK_POS_NUM				30
#define		MOVE_THRED					1.8f
#define		STATIC_THRED					0.8f

typedef enum{
	MOVE_UNKNOWN= 0x00,
	MOVE_A_STATIC	,
	MOVE_B_STATIC	,
	MOVE_AB_STATIC	,
	MOVE_POSITIVE	,
	MOVE_NEGATIVE	,
	MOVE_CRITICAL	,
}MOVE_MODE;

typedef enum{
	CUR_UN_KNOW	=		0x00,
	CUR_MOVE	,
	CUR_STATIC
}CUR_MV_TYPE;

typedef enum{
	LEFT_UP_BLOCK		= 	0x00,
	RIGHT_UP_BLOCK		,
	LEFT_BOT_BLOCK		,
	RIGTH_BOT_BLOCK	,
	MAX_SCENE_BLOCK
}SCENE_POS_TYPE;

typedef enum{
	REF_IMAGE_TYPE = 0x00,
	CUR_IMAGE_TYPE
}SCENE_REF_CUR_TYPE;

typedef struct _scene_mv_t{
	PointICR		mv[TRACK_POS_NUM];
	float 			conf[TRACK_POS_NUM];
	int 				num;
}SCENE_MV;

typedef struct _scene_param_t{
	UTC_Rect	refPos[MAX_SCENE_BLOCK];
	UTC_Rect	curPos[MAX_SCENE_BLOCK];
	UTC_Rect	trkPos[MAX_SCENE_BLOCK];
}SCENE_PARAM;

typedef struct _move_judge_t{
	int		history_num;
	int		track_num;
	float	move_thred;
	float	static_thred;
}MOVE_JUDGE_PARAM;

typedef	struct _traj_param_t{
	PointfCR			m_historyPos[HISTORY_POS_NUM*2];
	int					m_histNum;
	PointfCR			m_trackPos[TRACK_POS_NUM*2];
	int					m_trackNum;

	MOVE_JUDGE_PARAM	m_moveJudge;
	MOVE_MODE		m_moveType, m_mvBK;
	PointfCR				m_histDelta, m_trackDelta;
	int					m_intervNum;
	bool				m_bClear;
	float				m_trajAng;
}TRAJECTORY_PARAM;

typedef struct _scene_pos_confidence{
	PointICR		pos;
	float			opt;
}SCENE_POS_CONF;

typedef struct _scene_mv_obj_{
	TRAJECTORY_PARAM	m_trajParam;

	CUR_MV_TYPE				m_curMVMode;
	SCENE_MV						m_sceneMV;

	SCENE_PARAM				m_sceneParam;

	IMG_MAT						*_refSceneMap[MAX_SCENE_BLOCK];
	IMG_MAT						*_curSceneMap[MAX_SCENE_BLOCK];
	float 								refVar[MAX_SCENE_BLOCK];
	SCENE_POS_CONF		m_curPosConf[MAX_SCENE_BLOCK];

	SCENE_POS_CONF		m_curOptConf;
	bool								_bOptConf;
	int									m_width, m_height;

}SCENE_MV_OBJ, *SCENE_MV_HANDLE;

SCENE_MV_HANDLE		CreateSceneHdl();

void CloseSceneHdl(SCENE_MV_HANDLE pSceneMVObj);

void _initSceneMV(SCENE_MV_OBJ* pSceneMVObj, IMG_MAT *image);

void InitTrajectory(SCENE_MV_OBJ* pSceneMVObj);

void _unInitSceneMV(SCENE_MV_OBJ* pSceneMVObj);

int getSceneMap(SCENE_MV_OBJ* pSceneMVObj, IMG_MAT *image,  bool init);


void TrackJudge(SCENE_MV_OBJ* pSceneMVObj, float ratio);

void clrTrkState(SCENE_MV_OBJ* pSceneMVObj);

int getSceneMapEnh(SCENE_MV_OBJ* pSceneMVObj, IMG_MAT *image,  bool init, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit);

#endif
