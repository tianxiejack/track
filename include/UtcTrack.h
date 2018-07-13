#ifndef _UTC_TRK_
#define _UTC_TRK_

#include "PCTracker.h"


typedef enum{
	SEARCH_MODE_ALL	= 0x01,
	SEARCH_MODE_LEFT,
	SEARCH_MODE_RIGHT,
	SEARCH_MODE_NEAREST,
	SEARCH_MODE_MAX,
}SEARCH_MODE_TYPE;

typedef struct
{
	int  width;
	int  height;
}UTC_SIZE;

typedef struct
{
	int  x;
	int  y;
	int  width;
	int  height;
}UTC_Rect;

typedef struct
{
	float  x;
	float  y;
	float  width;
	float  height;
}UTC_RECT_float;

typedef struct _track_acq_param{
	int axisX;
	int axisY;
	UTC_Rect rcWin;
}UTC_ACQ_param;

typedef struct _utctrack_obj{
	bool     m_bInited;

	int axisX;
	int axisY;
	UTC_Rect rcWin;

}UTCTRACK_OBJ, *UTCTRACK_HANDLE;

typedef struct _utctrack_dynamic_param{
	float 	occlusion_thred;
	float	retry_acq_thred;

}UTC_DYN_PARAM;

typedef	struct	 _utctrack_search_param{
	int	min_bias_pix;
	int	max_bias_pix;
	float	blend_value;
}UTC_SEARCH_PARAM;

typedef enum{
	tPLT_TST = 0,
	tPLT_WRK = 1,
}tPLT;

typedef enum{
	BoreSight_Big	= 0,
	BoreSight_Mid	= 1,
	BoreSight_Sm	= 2,
}BS_Type;

typedef struct _trk_sech_t{
	int	res_distance;
	int	res_area;
}TRK_SECH_RESTRAINT;

UTCTRACK_HANDLE CreateUtcTrk();
void DestroyUtcTrk(UTCTRACK_HANDLE handle);

UTC_RECT_float UtcTrkAcq(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param inputParam);

/*UtcTrkAcqSR
 * bSalient: 使能目标大小自适应捕获
 * 作用=UtcTrkPreAcqSR+UtcTrkAcq
 */
UTC_RECT_float UtcTrkAcqSR(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param inputParam, bool bSalient);

UTC_RECT_float UtcTrkProc(UTCTRACK_OBJ *pUtcTrkObj, IMG_MAT frame, int *pRtnStat);

bool UtcTrkPreAcq(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param acqParam, UTC_Rect *pAcqRect, UTC_Rect *pCentRect,
									void  *pPatterns, int	*pNum);
bool UtcTrkPreAcq2(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param acqParam, UTC_Rect *pAcqRect, UTC_Rect *pAcqCentRect,
										UInt8*srMap,  UInt8*sobelMap, void  *pPatterns, int	*pNum);
/*UtcTrkPreAcqSR
 * 对给定区域(acqParam.rcWin指定)，进行目标检测提取
 * pAcqRect，返回距指定捕获中心(acqParam.rcWin的中心)的目标框
 * 返回值是true：表示找到目标，pAcqRect是找到的目标
 * 返回值是false：表示没有找到目标，pAcqRect是acqParam.rcWin
 */
bool UtcTrkPreAcqSR(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param acqParam, UTC_Rect *pAcqRect);

/*UtcBlobDetectSR
 * 对给定区域(acqParam.rcWin指定)，进行目标检测提取
 * pBlobRect，返回检测目标的目标框
 * 返回值是true：表示找到目标，pBlobRect是找到的目标
 * 返回值是false：表示没有找到目标，pBlobRect是acqParam.rcWin
 */
bool UtcBlobDetectSR(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param acqParam, UTC_Rect *pBlobRect);

bool UtcAcqTarget(UTCTRACK_OBJ* pUtcTrkObj, IMG_MAT frame, UTC_ACQ_param inputParam, UTC_Rect *pRect);

void UtcSetDynParam(UTCTRACK_OBJ *pUtcTrkObj, UTC_DYN_PARAM dynamicParam);
void UtcSetUpFactor(UTCTRACK_OBJ *pUtcTrkObj, float up_factor);
void UtcSetSerchMode(UTCTRACK_OBJ *pUtcTrkObj, SEARCH_MODE_TYPE searchMode);

void UtcGetOptValue(UTCTRACK_OBJ *pUtcTrkObj, float *optValue);
void UtcSetIntervalFrame(UTCTRACK_OBJ *pUtcTrkObj, int gapFrame);

void UtcSetSearchParam(UTCTRACK_OBJ *pUtcTrkObj, UTC_SEARCH_PARAM searchParam);
void UtcSetPLT_BS(UTCTRACK_OBJ *pUtcTrkObj, tPLT pltWork, BS_Type bsType);
/*set track search target size
 * default
 * resTrackObj={
	.res_distance = 80,
	.res_area = 5000,
};
 */
void UtcSetRestraint(UTCTRACK_OBJ *pUtcTrkObj,TRK_SECH_RESTRAINT resTraint);
/*
 * default disable enhance
 * bEnable = false;
 */
void UtcSetEnhance(UTCTRACK_OBJ *pUtcTrkObj, bool	bEnable);

/*if enable enhance, ifCliplimit will be efficient
 * fCliplimit = 4.0;
 */
void UtcSetEnhfClip(UTCTRACK_OBJ *pUtcTrkObj, float fCliplimit);
/*
 * default disable print timestamp
 * bPrint = false;
 */
void UtcSetPrintTS(UTCTRACK_OBJ *pUtcTrkObj, bool bPrint);
/*
 * default disable gaussfilter
 * bEnable = false;
 */
void UtcSetBlurFilter(UTCTRACK_OBJ *pUtcTrkObj, bool	bEnable);
/*
 * default
 * bPredict:false
 */
void UtcSetPredict(UTCTRACK_OBJ *pUtcTrkObj, bool bPredict);
/*
 * default
 * mvPixelX:20
 * mvPixelY:10
 */
void UtcSetMvPixel(UTCTRACK_OBJ *pUtcTrkObj, int mvPixelX, int mvPixelY);
/*
 * default
 * mvPixelX2:30
 * mvPixelY2:20
 */
void UtcSetMvPixel2(UTCTRACK_OBJ *pUtcTrkObj,  int mvPixelX2, int mvPixelY2);
/*
 * default
 * segPixelX:600
 * segPixelY:450
 */
void UtcSetSegPixelThred(UTCTRACK_OBJ *pUtcTrkObj,  int segPixelX, int segPixelY);
/*
 *default
 *maxValue = 0xFFFFFFF;
 **/
void UtcSetRoiMaxWidth(UTCTRACK_OBJ *pUtcTrkObj, int maxValue);

/*
 *default
 *thred = 40;Gray thred >T, white, ro black
*_salientThred = clip(10, 180, thred);
 **/
void UtcSetSalientThred(UTCTRACK_OBJ *pUtcTrkObj, int  thred);

/*
 *default
 *scatter = 10;
 *_salientScatter = clip(10, 200, scatter);
 **/
void UtcSetSalientScatter(UTCTRACK_OBJ *pUtcTrkObj, int scatter);

/*
 *default
 *large = 256;
 *mid = 128;
 *small = 64;
 *_smallScaler = clip(32, 192, small);
*	_midScaler = clip(96, 384, small);
*	_largeScaler = clip(128, 512, large);
*	设置大中小三个尺度的图像像素框大小
 **/
void UtcSetSalientScaler(UTCTRACK_OBJ *pUtcTrkObj, int large, int mid, int small);
/*
 * default
 * bMultScaler = 1
 * 设置目标检测多尺度或单尺度,1:mult scaler(large mid small scaler); 0: single scaer(large scaler)
 */
void UtcSetMultScaler(UTCTRACK_OBJ *pUtcTrkObj, int bMultScaler);
/*
 *default
 *minAcqSize.width = minAcqSize.height = 8
 *设置最小目标检测框的大小
 **/
void UtcSetSRMinAcqSize(UTCTRACK_OBJ *pUtcTrkObj, UTC_SIZE minAcqSize);

/*
 *default
 *ratio = 1.0
 *设置目标检测框大小放大倍数
 **/
void UtcSetSRAcqRatio(UTCTRACK_OBJ *pUtcTrkObj, float  ratio);
/*
 *default
 *bBigSearch = false
 *设置是否使能大范围搜索
 **/
void UtcSetBigSearch(UTCTRACK_OBJ *pUtcTrkObj, bool  bBigSearch);

/*
 *default
 *bDynamic = false
 *设置检测目标动态比例4:3 or 1:1
 **/
void UtcSetDynamicRatio(UTCTRACK_OBJ *pUtcTrkObj, bool  bDynamic);
/*
 *default
 *bSceneMV = false
 *设置场景运动使能
 **/
void UtcSetSceneMV(UTCTRACK_OBJ *pUtcTrkObj, bool  bSceneMV);
/*
 * default
 * bBackTrack = false
 * 设置使能回推
 */
void UtcSetBackTrack(UTCTRACK_OBJ *pUtcTrkObj, bool  bBackTrack);
/*
 * 得到场景运动速度
 */
void UtcGetSceneMV(UTCTRACK_OBJ *pUtcTrkObj, float *speedx, float *speedy);
/*
 * default
 * nThredFrms = 30
 * fRatioFrms = 0.25
 * 设置回推帧数阈值和比例阈值
 */
void UtcSetBackThred(UTCTRACK_OBJ *pUtcTrkObj, int  nThredFrms, float fRatioFrms);
/*
 *
 * bAveTrkPos = false;
 * 设置使能平均跟踪位置计算
 */
void UtcSetAveTrkPos(UTCTRACK_OBJ *pUtcTrkObj, bool  bAveTrkPos);
/*
 *default
 *fTau = 0.5
 *设置移动目标检测阴影阈值
 * Tau - shadow threshold. The shadow is detected if the pixel is darker
 *	version of the background. Tau is a threshold on how much darker the shadow can be.
 *	Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
 **/
void UtcSetDetectftau(UTCTRACK_OBJ *pUtcTrkObj, float  fTau);
/*
 *default
 *buildFrms = 500
 *设置移动目标检测模型建立帧数
 **/
void UtcSetDetectBuildFrms(UTCTRACK_OBJ *pUtcTrkObj, int  buildFrms);
/*
 * default
 * LostFrmThred = 30
 * 设置丢失检测帧数阈值
 */
void UtcSetLostFrmThred(UTCTRACK_OBJ *pUtcTrkObj, int  LostFrmThred);
/*
 * default
 * histMvThred = 1.0
 * 设置场景运动方向判定阈值
 */
void UtcSetHistMVThred(UTCTRACK_OBJ *pUtcTrkObj, float  histMvThred);
/*
 * default
 * detectFrms = 30
 * 设置运动检测静止判断帧数
 */
void UtcSetDetectFrmsThred(UTCTRACK_OBJ *pUtcTrkObj, int  detectFrms);
/*
 * default
 * detectFrms = 50
 * 设置静止等待帧数阈值
 */
void UtcSetStillFrmsThred(UTCTRACK_OBJ *pUtcTrkObj, int  stillFrms);
/*
 * default
 * stillThred = 0.1
 * 设置静止判断像素阈值
 */
void UtcSetStillPixThred(UTCTRACK_OBJ *pUtcTrkObj, float  stillThred);
/*
 * default
 * bKalmanFilter = false
 * 设置是否使能kalman滤波，UtcGetSceneMV()使能以后才起效
 */
void UtcSetKalmanFilter(UTCTRACK_OBJ *pUtcTrkObj, bool  bKalmanFilter);
/*
 * default
 * xMVThred = 3.0;
 * yMVThred = 2.0;
 * 设置kalman filter运动反向判断阈值
 */
void UtcSetKFMVThred(UTCTRACK_OBJ *pUtcTrkObj, float xMVThred, float yMVThred);
/*
 * default
 * xMVThred = 0.5;
 * yMVThred = 0.3;
 * 设置kalman filter静止状态判断阈值
 */
void UtcSetKFStillThred(UTCTRACK_OBJ *pUtcTrkObj, float xMVThred, float yMVThred);
/*
 * default
 * slopeThred = 0.08;
 * 设置Kalman Filter判断斜率阈值
 */
void UtcSetKFSlopeThred(UTCTRACK_OBJ *pUtcTrkObj, float slopeThred);
/*
 * default
 * kalmanHistThred = 2.5
 * 一般	kalmanHistThred<xMVThred||kalmanHistThred<yMVThred
 * 设置kalman filter历史轨迹大小使能判断阈值
 */
void UtcSetKFHistThred(UTCTRACK_OBJ *pUtcTrkObj, float kalmanHistThred);
/*
 *default
 * kalmanCoefQ = 0.00001
 * kalmanCoefR = 0.0025
 * 设置Kalman Filter 过程噪声协方差和测量噪声协方差误差值
 */
void UtcSetKFCoefQR(UTCTRACK_OBJ *pUtcTrkObj, float kalmanCoefQ, float kalmanCoefR);
/*
 *default
 * bSceneMVRecord = false;
 * 当需要记录场景运动时使能，一般使能bSceneMVRecord为true时，UtcSetSceneMV和UtcSetBackTrack设置为false
 */
void UtcSetSceneMVRecord(UTCTRACK_OBJ *pUtcTrkObj, bool  bSceneMVRecord);
/*
 *default
 *	trkAngleThred = 60.0;
 *	clip(30, 180, trkAngleThred)
 *设置跟踪搜索角度范围，<180 or >30
 */
void UtcSetTrkAngleThred(UTCTRACK_OBJ *pUtcTrkObj, float trkAngleThred);
/*
 * default
 * bHighOpt = false;
 * 搜索过程中使能高可信度判断
 */
void UtcSetTrkHighOpt(UTCTRACK_OBJ *pUtcTrkObj, bool bHighOpt);
/*
 * default
 * _bEnhScene = false
 * 使能场景区域增强
 */
void UtcSetSceneEnh(UTCTRACK_OBJ *pUtcTrkObj, bool bEnhScene);
/*
 * default
 * _nrxScene = 2;
 * _nrxScene = 2;
 * 设置场景区域增强宏块大小
 */
void UtcSetSceneEnhMacro(UTCTRACK_OBJ *pUtcTrkObj, unsigned int	 nrxScene, unsigned int nryScene);
/*
 *
 * 向算法通知私服水平和俯仰方向的速度，单位是像素
 */
void UtcNotifyServoSpeed(UTCTRACK_OBJ *pUtcTrkObj, float speedx, float speedy);
/*
 * default
 * peakMvThred = 1.0
 * 设置多峰检测运动方向判定阈值
 */
void UtcSetPeakMVThred(UTCTRACK_OBJ *pUtcTrkObj, float  peakMvThred);
#endif
