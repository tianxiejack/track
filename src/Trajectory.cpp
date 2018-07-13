#include <math.h>
#include "SceneMV.h"
#include "MatMem.h"

#define		SCENE_TRACK

static float _angleproc(PointfCR hist_delta, float hist_sqrt)
{
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

static void calTrajTrk(SCENE_MV_OBJ* pSceneMVObj)
{
	int i;
	TRAJECTORY_PARAM	*pTrajParam = &pSceneMVObj->m_trajParam;
	int	historyNum = pTrajParam->m_moveJudge.history_num;
	int	trackNum = pTrajParam->m_moveJudge.track_num;
	float	move_thred = pTrajParam->m_moveJudge.move_thred;
	float	static_thred = pTrajParam->m_moveJudge.static_thred;

	PointfCR	hist_delta,	track_delta;
	float	hist_sqrt, track_sqrt;
	float	hist_angle, track_angle;

	hist_delta.x	= hist_delta.y = 0.0f;
	track_delta.x = track_delta.y = 0.0f;
	pTrajParam->m_trajAng = 0.0;

	for(i=0; i<pTrajParam->m_histNum-pTrajParam->m_trackNum; i++)
	{
#ifndef	SCENE_TRACK
		hist_delta.x  += (pTrajParam->m_historyPos[i].x-pSceneMVObj->UtcTrkObj.axisX);
		hist_delta.y  += (pTrajParam->m_historyPos[i].y-pSceneMVObj->UtcTrkObj.axisY);
#else
		hist_delta.x  += pTrajParam->m_historyPos[i].x;
		hist_delta.y  += pTrajParam->m_historyPos[i].y;
#endif
	}
	hist_delta.x /= (pTrajParam->m_histNum-pTrajParam->m_trackNum);
	hist_delta.y /= (pTrajParam->m_histNum-pTrajParam->m_trackNum);
	hist_sqrt = sqrt(hist_delta.x*hist_delta.x+hist_delta.y*hist_delta.y);
	pTrajParam->m_histDelta  = hist_delta;

	for(i=0; i<pTrajParam->m_trackNum-1; i++)
	{
#ifndef	SCENE_TRACK
//		track_delta.x  += pTrajParam->m_trackPos[i+1].x - pTrajParam->m_trackPos[i].x;
//		track_delta.y  += pTrajParam->m_trackPos[i+1].y - pTrajParam->m_trackPos[i].y;
		track_delta.x  += (pTrajParam->m_trackPos[i].x-pDftTrkObj->UtcTrkObj.axisX);
		track_delta.y  += (pTrajParam->m_trackPos[i].y-pDftTrkObj->UtcTrkObj.axisY);
#else
		track_delta.x  += pTrajParam->m_trackPos[i].x;
		track_delta.y  += pTrajParam->m_trackPos[i].y;
#endif
	}
	track_delta.x /= (pTrajParam->m_trackNum-1);
	track_delta.y /= (pTrajParam->m_trackNum-1);
	track_sqrt = sqrt(track_delta.x*track_delta.x+track_delta.y*track_delta.y);
	pTrajParam->m_trackDelta  = track_delta;

	if(hist_sqrt < static_thred && track_sqrt < static_thred){
		pTrajParam->m_moveType = MOVE_AB_STATIC;
	}else if(hist_sqrt < static_thred && track_sqrt > move_thred){
		pTrajParam->m_moveType = MOVE_A_STATIC;
		pTrajParam->m_trajAng	= _angleproc(track_delta, track_sqrt);
	}else if(hist_sqrt > move_thred && track_sqrt < static_thred){
		pTrajParam->m_moveType = MOVE_B_STATIC;
	}else if(hist_sqrt > move_thred && track_sqrt > move_thred){
		hist_angle 	= _angleproc(hist_delta, hist_sqrt);
		track_angle	= _angleproc(track_delta, track_sqrt);
		pTrajParam->m_trajAng = track_angle;

		if(hist_angle>=0.0 && hist_angle<=90.0 && track_angle>=270.0 && track_angle<=360.0){
			hist_angle+= 360.0;
		}
		else if(track_angle>=0.0 && track_angle<=90.0 && hist_angle>=270.0 && hist_angle<=360.0){
			track_angle+= 360.0;
		}

		if(fabs(hist_angle-track_angle)>90.0){
			pTrajParam->m_moveType = MOVE_NEGATIVE;
		}else{
			pTrajParam->m_moveType = MOVE_POSITIVE;
		}
	}else if(hist_sqrt < move_thred && track_sqrt < move_thred && hist_sqrt > static_thred && track_sqrt > static_thred){//
		pTrajParam->m_moveType = MOVE_CRITICAL;
		pTrajParam->m_trajAng	= _angleproc(track_delta, track_sqrt);
	}else if(track_sqrt > move_thred ){
		pTrajParam->m_moveType = MOVE_CRITICAL;
		pTrajParam->m_trajAng	= _angleproc(track_delta, track_sqrt);
	}else if(track_sqrt > static_thred && track_sqrt < move_thred && hist_sqrt < static_thred){
		pTrajParam->m_moveType = MOVE_AB_STATIC;
	}
}

void TrackJudge(SCENE_MV_OBJ* pSceneMVObj, float ratio)
{
	TRAJECTORY_PARAM	*pTrajParam = &pSceneMVObj->m_trajParam;

	int i;
	int	historyNum = pTrajParam->m_moveJudge.history_num;
	int	trackNum = pTrajParam->m_moveJudge.track_num;
	float	move_thred = pTrajParam->m_moveJudge.move_thred;
	float	static_thred = pTrajParam->m_moveJudge.static_thred;
	PointfCR	hist_delta,	track_delta;

	if(!pSceneMVObj->_bOptConf)
		return;

	if(pTrajParam->m_histNum<historyNum)
	{
#ifndef	SCENE_TRACK
		pTrajParam->m_historyPos[pTrajParam->m_histNum].x = pDftTrkObj->_roi.x + pDftTrkObj->_roi.width/2;
		pTrajParam->m_historyPos[pTrajParam->m_histNum].y = pDftTrkObj->_roi.y + pDftTrkObj->_roi.height/2;
#else
		pTrajParam->m_historyPos[pTrajParam->m_histNum].x = pSceneMVObj->m_curOptConf.pos.x*ratio;
		pTrajParam->m_historyPos[pTrajParam->m_histNum].y = pSceneMVObj->m_curOptConf.pos.y*ratio;
#endif
		pTrajParam->m_histNum++;
	}
	else
	{
		memmove((void*)pTrajParam->m_historyPos, (void*)(pTrajParam->m_historyPos+1), (historyNum-1)*sizeof(PointfCR));
#ifndef	SCENE_TRACK
		pTrajParam->m_historyPos[historyNum-1].x = pDftTrkObj->_roi.x + pDftTrkObj->_roi.width/2;
		pTrajParam->m_historyPos[historyNum-1].y = pDftTrkObj->_roi.y + pDftTrkObj->_roi.height/2;
#else
		pTrajParam->m_historyPos[historyNum-1].x = pSceneMVObj->m_curOptConf.pos.x*ratio;
		pTrajParam->m_historyPos[historyNum-1].y = pSceneMVObj->m_curOptConf.pos.y*ratio;
#endif
	}

	if(pTrajParam->m_trackNum < trackNum)
	{
#ifndef	SCENE_TRACK
		pTrajParam->m_trackPos[pTrajParam->m_trackNum].x = pDftTrkObj->_roi.x + pDftTrkObj->_roi.width/2;
		pTrajParam->m_trackPos[pTrajParam->m_trackNum].y = pDftTrkObj->_roi.y + pDftTrkObj->_roi.height/2;
#else
		pTrajParam->m_trackPos[pTrajParam->m_trackNum].x = pSceneMVObj->m_curOptConf.pos.x*ratio;
		pTrajParam->m_trackPos[pTrajParam->m_trackNum].y = pSceneMVObj->m_curOptConf.pos.y*ratio;
#endif
		pTrajParam->m_trackNum++;
	}
	else
	{
		memmove((void*)pTrajParam->m_trackPos, (void*)(pTrajParam->m_trackPos+1), (trackNum-1)*sizeof(PointfCR));
#ifndef	SCENE_TRACK
		pTrajParam->m_trackPos[trackNum-1].x = pDftTrkObj->_roi.x + pDftTrkObj->_roi.width/2;
		pTrajParam->m_trackPos[trackNum-1].y = pDftTrkObj->_roi.y + pDftTrkObj->_roi.height/2;
#else
		pTrajParam->m_trackPos[trackNum-1].x = pSceneMVObj->m_curOptConf.pos.x*ratio;
		pTrajParam->m_trackPos[trackNum-1].y = pSceneMVObj->m_curOptConf.pos.y*ratio;
#endif
	}

	pTrajParam->m_moveType = MOVE_UNKNOWN;// initilize mv type

	if(pTrajParam->m_histNum >=trackNum*2 && pTrajParam->m_histNum <= historyNum && pTrajParam->m_trackNum==trackNum)
	{
		calTrajTrk(pSceneMVObj);
	}
	else
	{
		hist_delta.x = hist_delta.y = 0.f;
		track_delta.x = track_delta.y = 0.f;
		for(i=0; i<pTrajParam->m_histNum; i++){
			hist_delta.x += pTrajParam->m_historyPos[i].x;
			hist_delta.y += pTrajParam->m_historyPos[i].y;
		}
		for(i=0; i<pTrajParam->m_trackNum; i++){
			track_delta.x += pTrajParam->m_trackPos[i].x;
			track_delta.y += pTrajParam->m_trackPos[i].y;
		}

		pTrajParam->m_histDelta.x = hist_delta.x/pTrajParam->m_histNum;
		pTrajParam->m_histDelta.y = hist_delta.y/pTrajParam->m_histNum;
		pTrajParam->m_trackDelta.x = track_delta.x/pTrajParam->m_trackNum;
		pTrajParam->m_trackDelta.y = track_delta.y/pTrajParam->m_trackNum;
	}

	if(pTrajParam->m_moveType == MOVE_NEGATIVE && !pTrajParam->m_bClear){
		pTrajParam->m_intervNum = 0;
		pTrajParam->m_bClear = true;
	}

	if(pTrajParam->m_intervNum++ >= trackNum && pTrajParam->m_bClear){
		//	m_trackNum = 0;
		memmove(pTrajParam->m_historyPos, pTrajParam->m_historyPos+pTrajParam->m_histNum-trackNum-1, trackNum*sizeof(PointfCR));
		memcpy(pTrajParam->m_historyPos+trackNum, pTrajParam->m_trackPos, trackNum*sizeof(PointfCR));
		pTrajParam->m_histNum = trackNum*2;
		pTrajParam->m_bClear = false;
	}
}

void clrTrkState(SCENE_MV_OBJ* pSceneMVObj)
{
	TRAJECTORY_PARAM	*pTrajParam = &pSceneMVObj->m_trajParam;
	pTrajParam->m_trackNum = 0;
	pTrajParam->m_histNum = 0;
	pTrajParam->m_intervNum = 0;
	pTrajParam->m_bClear = false;
	pTrajParam->m_histDelta.x = pTrajParam->m_histDelta.y = 0.f;
	pTrajParam->m_trackDelta.x = pTrajParam->m_trackDelta.y = 0.f;
}

void InitTrajectory(SCENE_MV_OBJ* pSceneMVObj)
{
	TRAJECTORY_PARAM	*pTrajParam = &pSceneMVObj->m_trajParam;
	pTrajParam->m_histNum = 0;
	pTrajParam->m_trackNum = 0;
	pTrajParam->m_moveJudge.history_num = HISTORY_POS_NUM;
	pTrajParam->m_moveJudge.track_num = TRACK_POS_NUM;
	pTrajParam->m_moveJudge.move_thred = MOVE_THRED;
	pTrajParam->m_moveJudge.static_thred = STATIC_THRED;
	pTrajParam->m_moveType = MOVE_UNKNOWN;
	pTrajParam->m_mvBK = MOVE_UNKNOWN;
	pTrajParam->m_intervNum = 0;
	pTrajParam->m_bClear = false;
	pTrajParam->m_trajAng = 0.0;
}
