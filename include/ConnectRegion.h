#ifndef		_CONNECT_REGION_HEAD_
#define		_CONNECT_REGION_HEAD_

#include "PCTracker.h"
#include "UtcTrack.h"

#define SAMPLE_NUMBER	256

typedef	enum{
	CV_TYPE_UYVY		=		0x01,
	CV_TYPE_RGB		=		0x02,
	CV_TYPE_YUYV		=		0x03
}IMG_TYPE;

typedef struct _pattern_t //
{
	PointICR		lefttop;
	PointICR		rightbottom;//
	bool		   bValid;
	bool		   bEdge;
}Pattern;

typedef struct _connect_reg_t{
	Pattern  m_pPatterns[SAMPLE_NUMBER];
	int			m_patternnum; //
	UInt8     *m_ptemp;
	UInt8     *m_pBitData;
	int      m_dwWidth;
	int      m_dwHeight;
	int      m_iCount[SAMPLE_NUMBER];
	int      m_iRelative[SAMPLE_NUMBER];
	int      m_list[SAMPLE_NUMBER];//
}CON_REG_Obj, *CON_REG_HANDLE;

bool	CreatConRegObj(CON_REG_HANDLE *pConRegHdl);
void	DestroyRegObj(CON_REG_HANDLE *pConRegHdl);
bool   GetMoveDetect(CON_REG_HANDLE ConRegHdl, IMG_MAT src, UTC_Rect roi,  int T, int iscatter);
bool overlapRoi(UTC_Rect rec1,	UTC_Rect rec2, UTC_Rect *roi);

#endif
