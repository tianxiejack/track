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
#include "BackTrk.h"

static BACK_OBJ gBackObj;

BACK_HANDLE	CreatBackHandle()
{
	BACK_HANDLE	backHdl = (BACK_HANDLE)&gBackObj;
	memset(backHdl, 0, sizeof(BACK_HANDLE));
	for(int i=0; i<BACK_FRAMES_NUM; i++){
		backHdl->_histFrames[i] = NULL;
		backHdl->_histOpt[i] = 0.f;
	}
	backHdl->nFrames = 0;
	backHdl->nFrmIndx = 0;
	backHdl->nWidth = 0;
	backHdl->nHeight = 0;
	backHdl->_addSize.width = 60;
	backHdl->_addSize.height = 40;
	backHdl->_gapFrames = 1;
	backHdl->backStatus = 0;
	backHdl->_reAcqFlag = false;

	backHdl->_tmplateMap = NULL;
	backHdl->_curImgMap = NULL;
	return backHdl;
}

void	DestroyBackHandle(BACK_HANDLE backHdl)
{
	for(int i=0; i<BACK_FRAMES_NUM; i++){
		matFree(backHdl->_histFrames[i]);
		backHdl->_histFrames[i] = NULL;
		backHdl->_histOpt[i] = 0.f;
	}
	matFree(backHdl->_tmplateMap);
	matFree(backHdl->_curImgMap);
	backHdl->_tmplateMap = NULL;
	backHdl->_curImgMap = NULL;

	backHdl->nWidth = 0;
	backHdl->nHeight = 0;
	backHdl->nFrames = 0;
	backHdl->nFrmIndx = 0;
	memset(backHdl, 0, sizeof(BACK_HANDLE));
}

void initBackObj(BACK_OBJ *pBackObj, UTC_RECT_float roi, IMG_MAT *image)
{
	int i;
	UTILS_assert(roi.width >= 0 && roi.height >= 0);

	if(pBackObj->nWidth != image->width || pBackObj->nHeight != image->height){
		for(i=0; i<BACK_FRAMES_NUM; i++){
			matFree(pBackObj->_histFrames[i]);
			pBackObj->_histFrames[i] = NULL;
		}
		pBackObj->nWidth = image->width;
		pBackObj->nHeight = image->height;
		pBackObj->nFrames = 0;
		pBackObj->nFrmIndx = 0;
		pBackObj->_reAcqFlag = false;
	}
	if(pBackObj->_reAcqFlag){
		pBackObj->nFrames = 0;
		pBackObj->nFrmIndx = 0;
		pBackObj->_reAcqFlag = false;
	}

	pBackObj->_roi = roi;
	pBackObj->_curroi.x = roi.x - pBackObj->_addSize.width/2;
	pBackObj->_curroi.y =  roi.y - pBackObj->_addSize.height/2;
	pBackObj->_curroi.width = roi.width + pBackObj->_addSize.width;
	pBackObj->_curroi.height = roi.height + pBackObj->_addSize.height;

	for(i=0; i<BACK_FRAMES_NUM; i++){
		if(pBackObj->_histFrames[i] == NULL){
			pBackObj->_histFrames[i] = matAlloc(image->dtype,	image->width, image->height, image->channels);
		}
	}
}

void SetReBackFlag(HIST_TRACK_HANDLE pHistTrkObj)
{
	BACK_OBJ *pBackObj = (BACK_OBJ*)pHistTrkObj->m_backHdl;
	pBackObj->_reAcqFlag = true;
	pBackObj->nFrames = 0;
	pBackObj->nFrmIndx = 0;
}

void BackFrmRecord(HIST_TRACK_HANDLE pHistTrkObj, IMG_MAT frame, float opt, UTC_RECT_float utcRect, int utcStatus, PointfCR sceneVelocity)
{
	BACK_OBJ *pBackObj = (BACK_OBJ*)pHistTrkObj->m_backHdl;
	pBackObj->nFrames = (pBackObj->nFrames+1)>BACK_FRAMES_NUM?BACK_FRAMES_NUM:(pBackObj->nFrames+1);

//	matCopy(pBackObj->_histFrames[pBackObj->nFrmIndx], &frame);
	memcpy(pBackObj->_histFrames[pBackObj->nFrmIndx]->data_u8, frame.data_u8, frame.width*frame.height);
	pBackObj->_histOpt[pBackObj->nFrmIndx] = opt;
	pBackObj->_histRectPos[pBackObj->nFrmIndx] = utcRect;
	pBackObj->_histStatus[pBackObj->nFrmIndx] = utcStatus;
	pBackObj->sceneVelocity[pBackObj->nFrmIndx].x = sceneVelocity.x;
	pBackObj->sceneVelocity[pBackObj->nFrmIndx].y = sceneVelocity.y;
	pBackObj->nFrmIndx = (pBackObj->nFrmIndx+1) > (BACK_FRAMES_NUM-1) ? 0 : (pBackObj->nFrmIndx+1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static float calEuclidean(UTC_RECT_float fRect, UTC_RECT_float iRect)
{
	PointfCR pt1, pt2;
	pt1.x = fRect.x + fRect.width/2;		pt1.y = fRect.y + fRect.height/2;
	pt2.x = iRect.x + iRect.width/2;		pt2.y = iRect.y + iRect.height/2;

	return (float)(sqrt((pt1.x-pt2.x)*(pt1.x-pt2.x)+(pt1.y-pt2.y)*(pt1.y-pt2.y)));
}

static void _calVelocity(BACK_OBJ *pBackObj, PointfCR absDistance)
{
	float velocity ;
	if(absDistance.x < 2){
		pBackObj->_histVelocity.x = 2;
	}else if(absDistance.x < 4){
		pBackObj->_histVelocity.x = 4;
	}else if(absDistance.x < 6){
		pBackObj->_histVelocity.x = 6;
	}else if(absDistance.x < 8){
		pBackObj->_histVelocity.x = 8;
	}else if(absDistance.x < 10){
		pBackObj->_histVelocity.x = 10;
	}else if(absDistance.x < 14){
		pBackObj->_histVelocity.x = 14;
	}else if(absDistance.x < 20){
		pBackObj->_histVelocity.x = 20;
	}else if(absDistance.x < 24){
		pBackObj->_histVelocity.x = 24;
	}else{
		pBackObj->_histVelocity.x = 30;
	}

	if(absDistance.y < 2){
		pBackObj->_histVelocity.y = 2;
	}else if(absDistance.y < 4){
		pBackObj->_histVelocity.y = 4;
	}else if(absDistance.y < 6){
		pBackObj->_histVelocity.y = 6;
	}else if(absDistance.y < 8){
		pBackObj->_histVelocity.y = 8;
	}else if(absDistance.y < 10){
		pBackObj->_histVelocity.y = 10;
	}else if(absDistance.y < 14){
		pBackObj->_histVelocity.y = 14;
	}else if(absDistance.y < 20){
		pBackObj->_histVelocity.y = 20;
	}else if(absDistance.y < 24){
		pBackObj->_histVelocity.y = 24;
	}else{
		pBackObj->_histVelocity.y = 30;
	}

	velocity = sqrt(absDistance.x*absDistance.x+absDistance.y*absDistance.y);
	if(velocity <2){
		pBackObj->_gapFrames = 4;
	}else if(velocity <5){
		pBackObj->_gapFrames = 3;
	}else if(velocity <10){
		pBackObj->_gapFrames = 2;
	}else{
		pBackObj->_gapFrames = 1;
	}
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

UTC_RECT_float BackTrkProc(	HIST_TRACK_HANDLE pHistTrkObj, IMG_MAT frame, float opt, UTC_RECT_float utcRect, int utcStatus,
														PointfCR histVelocity, int *pRtnStat)
{
	UTC_RECT_float rcResult;
	PointfCR absDistance;
	int continueFrms = 0, nearestIdx, nearestIdx2;

	BACK_OBJ *pBackObj = (BACK_OBJ*)pHistTrkObj->m_backHdl;
	rcResult = pBackObj->_roi;

	pBackObj->backStatus = 1;//normal
	pBackObj->_bConf = true;

	absDistance.x = fabs(histVelocity.x);
	absDistance.y = fabs(histVelocity.y);
	_calVelocity(pBackObj, absDistance);

	if(pBackObj->nFrames >= BACK_FRAMES_NUM/2){
		int i, j, index;
		float	eulen, diagonal = sqrt(pBackObj->_roi.width*pBackObj->_roi.width + pBackObj->_roi.height*pBackObj->_roi.height);
		float	absV = sqrt(histVelocity.x*histVelocity.x+histVelocity.y*histVelocity.y);
		int      NFrames = (int)(diagonal*0.4/(absV+0.1));
		int      NFrames2 = (int)(diagonal*0.8/(absV+0.1));
		bool	bExe = true;
		float	fratio = 0.2;

		NFrames = NFrames<(pBackObj->nFrames/2-1)?NFrames:(pBackObj->nFrames/2-1);
		NFrames2 = NFrames2<(pBackObj->nFrames-2)?NFrames2:(pBackObj->nFrames-2);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		nearestIdx = NFrames;
		index = (pBackObj->nFrmIndx-1)<0?(pBackObj->nFrames-1) : (pBackObj->nFrmIndx-1);
		for(i=0; i<pBackObj->nFrames; i++){
			if(pBackObj->_histStatus[index] == 1){
				nearestIdx = i+1;
			}
			if((i+1) == NFrames)
				break;
			index = (index-1) < 0 ? (pBackObj->nFrames-1) : (index-1);
		}
		if(i == pBackObj->nFrames){
			bExe &= false;
		}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		nearestIdx2 = NFrames2;
		index = (pBackObj->nFrmIndx-1)<0?(pBackObj->nFrames-1) : (pBackObj->nFrmIndx-1);
		for(i=0; i<pBackObj->nFrames; i++){
			if(pBackObj->_histStatus[index] == 1){
				nearestIdx2 = i+1;
			}
			if((i+1) == NFrames2)
				break;
			index = (index-1) < 0 ? (pBackObj->nFrames-1) : (index-1);
		}
		if(i == pBackObj->nFrames){
			bExe &= false;
		}
/*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for(i=0,continueFrms = 0; i<nearestIdx2; i++){
			if(pBackObj->_histStatus[index] == 1){
				continueFrms++;
			}
			index = (index-1) < 0 ? (pBackObj->nFrames-1) : (index-1);
		}
		if(continueFrms <(nearestIdx2*0.65)){//trk status==1 frames is very small
			bExe &= false;
		}
		if(!bExe)
			pBackObj->_bConf = false;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/
		index = (pBackObj->nFrmIndx-1)<0?(pBackObj->nFrames-1) : (pBackObj->nFrmIndx-1);
		for(i=0,continueFrms=0; ((i<pBackObj->nFrames) && bExe); i++){
			continueFrms++;
			if(continueFrms == nearestIdx || continueFrms == nearestIdx2){
				if(pBackObj->_histStatus[index] != 1){
					index = (index-1) < 0 ? (pBackObj->nFrames-1) : (index-1);
					continue;
				}
/**********************************************************************************/
				//pHistTrkObj->_roi = pBackObj->_histRectPos[index];//*******************//
/**********************************************************************************/
				HistTrkProc(pHistTrkObj, *pBackObj->_histFrames[index], NULL, false);

				if(pHistTrkObj->utcStatus == pBackObj->_histStatus[index]){
					pBackObj->trkPosOpt[i] = pHistTrkObj->opt_peak_value;
					pBackObj->trkPos[i] = pHistTrkObj->_roi;

					eulen = calEuclidean(pBackObj->_histRectPos[index], pBackObj->trkPos[i]);
					if(continueFrms == nearestIdx){
						fratio = 0.15;
					}else if(continueFrms == nearestIdx2){
						fratio = 0.2;
					}
					if(eulen < (diagonal*fratio)){
						pBackObj->backStatus = 1;//normal
						pBackObj->_bConf = true;
					}else{
						pBackObj->backStatus = 0;//abnormal
						pBackObj->_bConf = false;
						printf("%s:Back Trk is abnormal! eulen > diagonal*%0.2f\n",__func__, fratio);
						break;
					}
				}else{
					pBackObj->backStatus = 0;//abnormal
					pBackObj->_bConf = false;
					printf("%s:Back Trk is abnormal! utcStatus = %d\n",__func__, pHistTrkObj->utcStatus);
					break;
				}
			}
			index = (index-1) < 0 ? (pBackObj->nFrames-1) : (index-1);
		}
	}

	BackFrmRecord(pHistTrkObj, frame, opt, utcRect, utcStatus, histVelocity);

	return rcResult;
}


