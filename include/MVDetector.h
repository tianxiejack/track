#ifndef		_MV_DETECTOR_HEAD_
#define		_MV_DETECTOR_HEAD_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "MOG3.hpp"

using namespace cv;
using namespace std;
using namespace OurMogBgs;

#define		DETECTOR_NUM		2
#define		BGFG_CR
//#undef		BGFG_CR

typedef struct _mog2_para_t{
	bool	bShadowDetection;//阴影去除，默认打开
	 int		history;//背景建模历史帧数(背景模型影响帧数)，默认为500,	Learning rate; alpha = 1/defaultHistory2
	 double varThreshold;//模板匹配阈值4.0f*4.0f
	 float	fTau;//阴影阈值0.5f
	 float	varThresholdGen;//新模型匹配阈值3.0f*3.0f;
	 unsigned char nShadowDetection;//前景中模型参数，设置为0表示背景，255表示前景，默认值为127
																		//do shadow detection - insert this value as the detection result - 127 default value

	 float	backgroundRatio;//背景阈值设定:0.9f; backgroundRatio*history; // threshold sum of weights for background test
	 int		nmixtures;//高斯混合模型组件数量5; // maximal number of Gaussians in mixture
	 float	fVarInit;//每个高斯组件的初始方差15.0f; // initial variance for new components
	 float	fVarMin;//4.0f
	 float	fVarMax;//5*defaultVarInit2;

	 float fCT;//CT - complexity reduction prior:0.05f

}MOG2_PARAM;

class CMoveDetector
{
public:
	CMoveDetector();
	virtual	~CMoveDetector();

	int		creat(int history = 500,  float varThreshold = 16, bool bShadowDetection=true);
	int		destroy();
	int		init();
	void 	DetectProcess(cv::Mat	frame, int chId = 0);

#ifdef		BGFG_CR
	void	setDetectShadows(bool	bShadow,	int chId	= 0);
	void	setHistory(int nHistory,	int chId	= 0);
	void	setVarThreshold(double varThreshold,	int chId	= 0);
	void	setVarThredGen(float	varThredGen,	int chId	= 0);
	void	setBackgroundRatio(float ratio,	int chId	= 0);
	void	setCompRedThred(float fct,	int chId	= 0);
	void	setNMixtures(int nmix,	int chId	= 0);
	void	setVarInit(float initvalue,	int chId	= 0);
	void	setShadowValue(int value,	int chId	= 0);
	void	setShadowThreshold(double threshold,	int chId	= 0);
	void	setNFrames(int nframes, int chId = 0);
#endif

public:
	MOG2_PARAM	m_mogParam;
	cv::Mat	frame[DETECTOR_NUM];
	cv::Mat 	fgmask[DETECTOR_NUM];


private:

#ifdef		BGFG_CR
		BackgroundSubtractorMOG3* fgbg[DETECTOR_NUM];
#else
		BackgroundSubtractorMOG2 *fgbg[DETECTOR_NUM];
#endif

};

#endif
