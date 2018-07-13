#include "MVDetector.h"
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/video/background_segm.hpp>

CMoveDetector::CMoveDetector()
{
	int	i;
	for(i=0; i<DETECTOR_NUM; i++){
#ifdef		BGFG_CR
		fgbg[i] = NULL;
#else
		fgbg[i] = NULL;
#endif
	}
}

CMoveDetector::~CMoveDetector()
{
	destroy();
}

int CMoveDetector::creat(int history /*= 500*/,  float varThreshold /*= 16*/, bool bShadowDetection /*=true*/)
{
	int	i;
	initModule_video();
	setUseOptimized(true);
	setNumThreads(4);

	for(i=0; i<DETECTOR_NUM; i++)
	{
#ifdef	BGFG_CR
		if(fgbg[i] != NULL){
			delete fgbg[i];
			fgbg[i] = NULL;
		}
		fgbg[i] = new BackgroundSubtractorMOG3;
		if(fgbg[i] == NULL)
		{
			printf( "%s:Failed to create BackgroundSubtractor.MOG3 Algorithm.",__func__ );
			assert(0);
		}
#else
//		fgbg[i] = Algorithm::create<BackgroundSubtractorMOG2>("BackgroundSubtractor.MOG2");
		if(fgbg[i] != NULL){
			delete fgbg[i];
			fgbg[i] = NULL;
		}
		fgbg[i] = new BackgroundSubtractorMOG2(history, varThreshold, bShadowDetection);
		if (fgbg[i] == NULL)
		{
			printf( "Failed to create BackgroundSubtractor.MOG2 Algorithm." );
			assert(0);
		}
#endif
	}

	return	0;
}

int CMoveDetector::init()
{
	return 0;
}

int CMoveDetector::destroy()
{
	int	i,	rtn;

	for(i=0; i<DETECTOR_NUM; i++)
	{
#ifdef		BGFG_CR
		if(fgbg[i] != NULL)
		{
			delete fgbg[i];
			fgbg[i] = NULL;
		}
#else
		if(fgbg[i] != NULL)
		{
			delete fgbg[i];
			fgbg[i] = NULL;
		}
#endif
	}

	return rtn;
}

#ifdef		BGFG_CR
void	CMoveDetector::setDetectShadows(bool	bShadow,	int chId/*	= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setDetectShadows( bShadow);
	}
}

void	CMoveDetector::setShadowValue(int value,	int chId	/*= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setShadowValue( value);
	}
}

void	CMoveDetector::setHistory(int nHistory,	int chId/*	= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setHistory(nHistory);
	}
}

void	CMoveDetector::setVarThreshold(double varThreshold,	int chId/*	= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setVarThreshold(varThreshold);
	}
}

void	CMoveDetector::setVarThredGen(float	varThredGen,	int chId/*	= 0*/)
{

	if (fgbg[chId] != NULL){
		fgbg[chId]->setVarThresholdGen(varThredGen);
	}
}

void	CMoveDetector::setBackgroundRatio(float ratio,	int chId/*	= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setBackgroundRatio(ratio);
	}
}

void	CMoveDetector::setCompRedThred(float fct,	int chId/*	= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setComplexityReductionThreshold(fct);
	}
}

void	CMoveDetector::setNMixtures(int nmix,	int chId/*	= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setNMixtures(nmix);
	}
}

void	CMoveDetector::setVarInit(float initvalue,	int chId/*	= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setVarInit(initvalue);
	}
}

void	CMoveDetector::setShadowThreshold(double threshold,	int chId	/*= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setShadowThreshold(threshold);
	}
}

void	CMoveDetector::setNFrames(int nframes, int chId /*= 0*/)
{
	if (fgbg[chId] != NULL){
		fgbg[chId]->setDetectNFrames(nframes);
	}
}

#endif

void CMoveDetector::DetectProcess(cv::Mat	frame, int chId /*= 0*/)
{
	CV_Assert(chId < DETECTOR_NUM);
	bool update_bg_model = true;
	if(!frame.empty())
	{
		(*fgbg[chId])(frame, fgmask[chId], update_bg_model ? -1 : 0);
	}

}
