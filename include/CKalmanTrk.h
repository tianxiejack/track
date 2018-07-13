#ifndef		_KALMAN_TRK_HEAD_
#define		_KALMAN_TRK_HEAD_

#include "Kalman.h"
#include "DFT.h"

#define		RECORD_FILTER_NUM		90

class	CKalmanTracker{
public:
	CKalmanTracker();
	~CKalmanTracker();

public:
	void	InitKalmanTrk(PointfCR	trkPoint, double DeltaT,int DP, int MP, int CP, float Q, float R);
	void 	unInitKalmanTrk();

	void Kalman(double *measure, double *control);
	void KalmanPredict(int xout, int yout);

	void KalmanTrkAcq(PointfCR	trkPoint, float Q, float R);
	void KalmanTrkPredict(PointfCR	&trkPoint);
	void KalmanTrkProc(PointfCR	trkPoint);
	void KalmanTrkFilter(PointfCR	trkPoint, float &deltax, float &deltay);
	int KalmanTrkJudge(PointfCR	KalmanMVThred, PointfCR KalmanStillThred, float slopeThred, PointfCR *cstSpeed, bool bSceneMVRecord);
	void KalmanTrkReset();

public:
	CKalman* m_pKalmanProc;
	double*  m_pMeasure;
	double*  m_pControl;
	PointfCR	  m_trackParam;
	PointfCR	  m_filterRecord[RECORD_FILTER_NUM];
	int			  m_nRecord;

	bool	m_bInited;
};


#endif
