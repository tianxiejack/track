#include "CKalmanTrk.h"

CKalmanTracker::CKalmanTracker():m_pKalmanProc(NULL),m_pMeasure(NULL),m_pControl(NULL)
{
	m_bInited = false;
	m_nRecord = 0;
}

CKalmanTracker::~CKalmanTracker()
{
	unInitKalmanTrk();
}

void CKalmanTracker::Kalman(double *measure, double *control)
{
	m_pKalmanProc->KalmanPredict(control);
	m_pKalmanProc->KalmanCorrect(measure);
}

void CKalmanTracker::KalmanPredict(int xout, int yout)
{
	m_pMeasure[0] = (double)xout;
	m_pMeasure[1] = (double)yout;
	Kalman(m_pMeasure,	NULL);
}

void CKalmanTracker::unInitKalmanTrk()
{
	if (m_pKalmanProc != NULL)
	{
		delete m_pKalmanProc;
		m_pKalmanProc = NULL;
	}
	if (m_pMeasure != NULL)
	{
		delete m_pMeasure;
		m_pMeasure = NULL;
	}
	if (m_pControl != NULL)
	{
		delete m_pControl;
		m_pControl = NULL;
	}
	m_bInited = FALSE;
	m_nRecord = 0;
}

void	CKalmanTracker::InitKalmanTrk(PointfCR	trkPoint, double DeltaT,int DP, int MP, int CP, float Q, float R)
{
	float x0, y0;
	if (m_bInited){
		unInitKalmanTrk();
	}
	if (m_pKalmanProc == NULL){
		m_pKalmanProc = new CKalman();
		if ( m_pKalmanProc == NULL){
			m_bInited = FALSE;
			return;
		}
	}
	m_pKalmanProc->KalmanOpen(DP, MP, CP);
	if (!m_pKalmanProc->m_bInited){
		m_bInited = FALSE;
		return;
	}
	x0 = trkPoint.x ;
	y0 = trkPoint.y ;
	m_pKalmanProc->KalmanInitParam((double)x0, (double)y0, DeltaT, (double)Q, (double)R);

	if (m_pMeasure == NULL){
		m_pMeasure = new double[MP * 1];
		memset(m_pMeasure, 0, sizeof(double)*MP);
	}
	if (m_pControl == NULL && CP > 0){
		m_pControl = new double[CP * 1];
		memset(m_pControl, 0, sizeof(double)*CP);
	}
	m_bInited = TRUE;
	m_nRecord = 0;
}

void CKalmanTracker::KalmanTrkPredict(PointfCR	&trkPoint)
{
	float xin, yin, xin1, yin1;
	xin = m_trackParam.x;
	yin = m_trackParam.y;
	xin1 =  xin + m_pKalmanProc->state_post[2] *m_pKalmanProc->deltat;//Kalman filter
	yin1 =  yin + m_pKalmanProc->state_post[3] *m_pKalmanProc->deltat;
	trkPoint.x = m_trackParam.x	+	xin1;
	trkPoint.y = m_trackParam.y	+	yin1;
}

void CKalmanTracker::KalmanTrkProc(PointfCR	trkPoint)
{
	m_trackParam = trkPoint;
	m_pMeasure[0] = (double)(trkPoint.x);
	m_pMeasure[1] = (double)(trkPoint.y);
	Kalman(m_pMeasure,	NULL);
}

void	CKalmanTracker::KalmanTrkAcq(PointfCR	trkPoint, float Q, float R)
{
	double delta = 1/40.0;
	int DP =	4, MP = 2, CP = 0;
	InitKalmanTrk(trkPoint, delta, DP, MP, CP, Q, R);
	m_trackParam = trkPoint;
}

#if 0
void CKalmanTracker::KalmanTrkFilter(PointfCR	trkPoint, float &deltax, float &deltay)
{
	m_trackParam.x += trkPoint.x;
	m_trackParam.y += trkPoint.y;
	m_pMeasure[0] = (double)(m_trackParam.x);
	m_pMeasure[1] = (double)(m_trackParam.y);
	Kalman(m_pMeasure,	NULL);
	deltax       = (float)( m_pKalmanProc->state_post[0] - m_trackParam.x);
	deltay       = (float)( m_pKalmanProc->state_post[1] - m_trackParam.y);
}
#else
void CKalmanTracker::KalmanTrkFilter(PointfCR	trkPoint, float &deltax, float &deltay)
{
	m_trackParam.x = trkPoint.x;
	m_trackParam.y = trkPoint.y;
	m_pMeasure[0] = (double)(m_trackParam.x);
	m_pMeasure[1] = (double)(m_trackParam.y);
	Kalman(m_pMeasure,	NULL);
	deltax       = (float)( m_pKalmanProc->state_post[0] - m_trackParam.x);
	deltay       = (float)( m_pKalmanProc->state_post[1] - m_trackParam.y);

	if(m_nRecord<RECORD_FILTER_NUM){
		m_filterRecord[m_nRecord].x = (float)( m_pKalmanProc->state_post[0]);
		m_filterRecord[m_nRecord].y = (float)( m_pKalmanProc->state_post[1]);
		m_nRecord++;
	}else{
		memmove(m_filterRecord, (m_filterRecord+1), sizeof(PointfCR)*(m_nRecord-1));
		m_filterRecord[m_nRecord-1].x = (float)( m_pKalmanProc->state_post[0]);
		m_filterRecord[m_nRecord-1].y = (float)( m_pKalmanProc->state_post[1]);
	}
}
#endif

int CKalmanTracker::KalmanTrkJudge(PointfCR	KalmanMVThred, PointfCR KalmanStillThred,  float slopeThred, PointfCR *cstSpeed, bool bSceneMVRecord)
{
	int i, j;
	float slope;
	if(m_nRecord >= (RECORD_FILTER_NUM*2/3))
	{
		if( fabs(m_filterRecord[0].x)>KalmanMVThred.x && fabs(m_filterRecord[m_nRecord-1].x)<KalmanStillThred.x )
		{
			for(j=m_nRecord-1; j>=0; j--){
				if( fabs(m_filterRecord[0].x-m_filterRecord[j].x) < KalmanStillThred.x){
					break;
				}
			}
			slope = (m_filterRecord[m_nRecord-1].x-m_filterRecord[j].x)/(m_nRecord-j);
			if(fabs(slope)> slopeThred){
				if(cstSpeed != NULL){
					cstSpeed->x = m_filterRecord[0].x;
					cstSpeed->y = m_filterRecord[0].y;
				}
				if(bSceneMVRecord){
					printf("%s:m_filterRecord[m_nRecord-1].x=%0.2f, m_filterRecord[j].x=%0.2f, slope=%f, m_nRecord-j=%d \n",__func__,
										m_filterRecord[m_nRecord-1].x, m_filterRecord[j].x, slope, m_nRecord-j);
				}
				return 1;
			}
		}
		else if( fabs(m_filterRecord[0].y)>KalmanMVThred.y && fabs(m_filterRecord[m_nRecord-1].y)<KalmanStillThred.y )
		{
			for(j=m_nRecord-1; j>=0; j--){
				if( fabs(m_filterRecord[0].y-m_filterRecord[j].y) < KalmanStillThred.y){
					break;
				}
			}
			slope = (m_filterRecord[m_nRecord-1].y-m_filterRecord[j].y)/(m_nRecord-j);
			if(fabs(slope)> slopeThred){
				if(cstSpeed != NULL){
					cstSpeed->x = m_filterRecord[0].x;
					cstSpeed->y = m_filterRecord[0].y;
				}
				if(bSceneMVRecord){
					printf("%s:m_filterRecord[m_nRecord-1].y=%0.2f, m_filterRecord[j].y=%0.2f, slope=%f, m_nRecord-j=%d \n",__func__,
							 	 	 	 m_filterRecord[m_nRecord-1].y, m_filterRecord[j].y, slope, m_nRecord-j);
				}
				return 2;
			}
		}
		else if( fabs(m_filterRecord[0].x)<KalmanStillThred.x && fabs(m_filterRecord[0].y)<KalmanStillThred.y )//still status
		{
			if(fabs(m_filterRecord[m_nRecord-1].x)>KalmanMVThred.x ){
				for(j=m_nRecord-1; j>=0; j--){
					if( fabs(m_filterRecord[0].x-m_filterRecord[j].x) < KalmanStillThred.x){
						break;
					}
				}
				slope = (m_filterRecord[m_nRecord-1].x-m_filterRecord[j].x)/(m_nRecord-j);
				if(fabs(slope)> slopeThred){
					if(cstSpeed != NULL){
						cstSpeed->x = -m_filterRecord[m_nRecord-1].x;
						cstSpeed->y = -m_filterRecord[m_nRecord-1].y;
					}
					if(bSceneMVRecord){
						printf("%s:m_filterRecord[m_nRecord-1].x=%0.2f, m_filterRecord[j].x=%0.2f, slope=%f, m_nRecord-j=%d \n",__func__,
											m_filterRecord[m_nRecord-1].x, m_filterRecord[j].x, slope, m_nRecord-j);
					}
					return 3;
				}
			}else if(fabs(m_filterRecord[m_nRecord-1].y)>KalmanMVThred.y){
				for(j=m_nRecord-1; j>=0; j--){
					if( fabs(m_filterRecord[0].y-m_filterRecord[j].y) < KalmanStillThred.y){
						break;
					}
				}
				slope = (m_filterRecord[m_nRecord-1].y-m_filterRecord[j].y)/(m_nRecord-j);
				if(fabs(slope)> slopeThred){
					if(cstSpeed != NULL){
						cstSpeed->x = -m_filterRecord[m_nRecord-1].x;
						cstSpeed->y = -m_filterRecord[m_nRecord-1].y;
					}
					if(bSceneMVRecord){
						printf("%s:m_filterRecord[m_nRecord-1].y=%0.2f, m_filterRecord[j].y=%0.2f, slope=%f, m_nRecord-j=%d \n",__func__,
											m_filterRecord[m_nRecord-1].y, m_filterRecord[j].y, slope, m_nRecord-j);
					}
					return 4;
				}
			}
		}
	}

	return 0;
}

void CKalmanTracker::KalmanTrkReset()
{
	m_nRecord = 0;
}
