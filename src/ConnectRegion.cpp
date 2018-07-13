#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "DFT.h"
#include "ConnectRegion.h"

static bool    InitializedMD(CON_REG_HANDLE ConRegHdl, int lWidth, int lHeight, int lStride);
static void	DestroyMD(CON_REG_HANDLE ConRegHdl);
static void	MergeRect(CON_REG_HANDLE ConRegHdl,  Pattern	ptn[], int num);

bool overlapRoi(UTC_Rect rec1,	UTC_Rect rec2, UTC_Rect *roi)
{
	PointICR tl1, tl2;
	UTC_SIZE sz1,sz2;
	int x_tl, y_tl, x_br, y_br;
	tl1.x	= rec1.x;	tl1.y	= rec1.y;
	tl2.x	= rec2.x;	tl2.y	= rec2.y;
	sz1.width	= rec1.width;	sz1.height	= rec1.height;
	sz2.width	= rec2.width;	sz2.height	= rec2.height;

	x_tl = MAX(tl1.x, tl2.x);
	y_tl = MAX(tl1.y, tl2.y);
	x_br = MIN(tl1.x + sz1.width, tl2.x + sz2.width);
	y_br = MIN(tl1.y + sz1.height, tl2.y + sz2.height);
	if (x_tl < x_br && y_tl < y_br)
	{
		roi->x = x_tl;	roi->y = y_tl;	roi->width = x_br - x_tl;	roi->height = y_br - y_tl;	
		return true;
	}
	return false;
}

static int _bInRect(UTC_Rect rec1,	UTC_Rect	rec2,	UTC_Rect *roi)
{
	bool rtn	= overlapRoi(rec1, rec2,	 roi);
	if(rtn){
		if(roi->x == rec1.x && roi->y == rec1.y&& roi->width == rec1.width&& roi->height == rec1.height)
			return 1;
		else if(roi->x == rec2.x && roi->y == rec2.y&& roi->width == rec2.width&& roi->height == rec2.height)
			return 2;
		else
			return	0;
	}else{
		return -1;
	}
}

static float _bbOverlap(const UTC_Rect box1,const UTC_Rect box2)
{
	float colInt, rowInt, intersection, area1, area2;
	if (box1.x > box2.x+box2.width) { return 0.0; }
	if (box1.y > box2.y+box2.height) { return 0.0; }
	if (box1.x+box1.width < box2.x) { return 0.0; }
	if (box1.y+box1.height < box2.y) { return 0.0; }

	 colInt =  MIN(box1.x+box1.width,box2.x+box2.width) - MAX(box1.x, box2.x);
	rowInt =  MIN(box1.y+box1.height,box2.y+box2.height) - MAX(box1.y,box2.y);

	intersection = colInt * rowInt;
	area1 = box1.width*box1.height;
	area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

bool	CreatConRegObj(CON_REG_HANDLE *pConRegHdl)
{
	CON_REG_HANDLE	ConRegHdl = (CON_REG_HANDLE)malloc(sizeof(CON_REG_Obj));
	if(ConRegHdl != NULL && pConRegHdl != NULL){
		ConRegHdl->m_dwWidth = 0;
		ConRegHdl->m_dwHeight = 0;
		ConRegHdl->m_patternnum = 0;
		ConRegHdl->m_ptemp = NULL;
		ConRegHdl->m_pBitData = NULL;
		*pConRegHdl = ConRegHdl;
		return true;
	}else{
		return false;
	}
}

void	DestroyRegObj(CON_REG_HANDLE *pConRegHdl)
{
	if(*pConRegHdl == NULL)
		return;
	DestroyMD(*pConRegHdl);
	if(*pConRegHdl != NULL){
		free(*pConRegHdl);
		*pConRegHdl = NULL;
	}
}

static void	DestroyMD(CON_REG_HANDLE ConRegHdl)
{
	if (ConRegHdl->m_ptemp != NULL)
		free(ConRegHdl->m_ptemp);
	if(ConRegHdl->m_pBitData != NULL)
		free(ConRegHdl->m_pBitData);
	ConRegHdl->m_ptemp = NULL;
	ConRegHdl->m_pBitData = NULL;
	ConRegHdl->m_dwWidth = 0;
	ConRegHdl->m_dwHeight = 0;
}

static bool    InitializedMD(CON_REG_HANDLE ConRegHdl, int lWidth, int lHeight, int lStride)
{
	if (ConRegHdl->m_dwHeight != lHeight || ConRegHdl->m_dwWidth != lWidth)
		DestroyMD(ConRegHdl);

	if (ConRegHdl->m_ptemp == NULL){
		ConRegHdl->m_ptemp = (UInt8*)malloc(lWidth*lHeight*sizeof(UInt8));
		if (ConRegHdl->m_ptemp == NULL){
			return false;
		}
	}

	if(ConRegHdl->m_pBitData == NULL){
		ConRegHdl->m_pBitData = (UInt8*)malloc(lStride * lHeight*sizeof(UInt8));
		if (ConRegHdl->m_pBitData == NULL){
			free(ConRegHdl->m_ptemp);
			return false;
		}
	}

	ConRegHdl->m_dwWidth = lWidth;
	ConRegHdl->m_dwHeight = lHeight;	
	return true;
}

/*
GetMoveDetect(LPBYTE lpBitData,int lWidth, int lHeight, int iStride, int iscatter  = 5)
lpBitData :
iscatter:
m_patternnum
m_pPattern[]:
 */
#define BOL		16
#if 0
bool    GetMoveDetect(CON_REG_HANDLE ConRegHdl, IMG_MAT src, UTC_Rect roi, int iscatter)
{
	bool iRet = true;
	int lWidth, lHeight, iStride;
	UInt8 *lpBitData = (UInt8 *)src.data_u8;

	int i, j, t, k;
	int label = 1; 
	int kx, ky;
	int minlabel=-1, lab[5]; //label
	int curlab=-1;
	int max=0;

	const int T = 60;
	bool	bJmp = false;
	Pattern ptn[SAMPLE_NUMBER];
	int patternnum, iCur;

	lWidth = src.width;	lHeight = src.height;	 iStride = src.step[0];
	iRet = InitializedMD(ConRegHdl, lWidth, lHeight, iStride);
	if (!iRet) return false;

	memset(&ptn, 0, sizeof(ptn));
	memset(ConRegHdl->m_ptemp, 0, lWidth*lHeight*sizeof(UInt8));
	memset(ConRegHdl->m_iCount,0,sizeof(int)*SAMPLE_NUMBER);
	memset(ConRegHdl->m_iRelative,0,sizeof(int)*SAMPLE_NUMBER);
	memset(ConRegHdl->m_list,0,sizeof(int)*SAMPLE_NUMBER);

	memset(lpBitData, 0, iStride*BOL);
	memset(lpBitData+iStride*(lHeight-BOL), 0, iStride*BOL);
	for(iCur=0; iCur<lHeight; iCur++){
		memset(lpBitData+iCur*iStride, 0, BOL);
		memset(lpBitData+iCur*iStride + lWidth - BOL, 0, BOL);
	}

	ConRegHdl->m_list[0] = -1;

	for( j=BOL; j<lHeight-BOL	&& !bJmp; j++)
	{
		for( i=BOL; i<lWidth-BOL	&& !bJmp; i++)
		{
			if(*(lpBitData + j*iStride + i) > T)
			{
				minlabel=-1;
				lab[1]=-1; lab[2]=-1; lab[3]=-1; lab[4]=-1;

				ky=j-1; kx=i+1;
				if(*(lpBitData + ky*iStride + kx)>T)
				{
					lab[1]= ConRegHdl->m_ptemp[ky*lWidth + kx];
					minlabel = ConRegHdl->m_ptemp[ky*lWidth + kx];
				}

				ky=j-1; kx=i;
				if(*(lpBitData + ky*iStride + kx)>T)
				{
					lab[2] = ConRegHdl->m_ptemp[ky*lWidth + kx];
					if(minlabel>lab[2]||minlabel==-1)
					{
						minlabel=lab[2];
					}
				}

				ky=j-1; kx=i-1;
				if(*(lpBitData + ky*iStride + kx)>T)
				{
					lab[3] = ConRegHdl->m_ptemp[ky*lWidth + kx];
					if(minlabel>lab[3]||minlabel==-1)
					{
						minlabel=lab[3];
					}
				}

				ky=j; kx=i-1;
				if(*(lpBitData + ky*iStride + kx)>T)
				{
					lab[4] = ConRegHdl->m_ptemp[ky*lWidth + kx];
					if(minlabel>lab[4]||minlabel==-1)
					{
						minlabel=lab[4];
					}
				}
				if(minlabel<0)//
				{
					ConRegHdl->m_ptemp[j*lWidth + i]=label;
					ConRegHdl->m_list[label]=-1;
					label++;
				}
				else//minlabel>=0
				{
					if(lab[1]==lab[2] && lab[2]==lab[3] && lab[3]==lab[4])
					{
						ConRegHdl->m_ptemp[j*lWidth + i] = minlabel;
					}
					else if(minlabel==(lab[1]+lab[2]+lab[3]+lab[4]+3))
					{
						ConRegHdl->m_ptemp[j*lWidth + i] = minlabel;
					}
					else
					{
						ConRegHdl->m_ptemp[j*lWidth + i] = minlabel;
						for( k=1; k<=4; k++)
						{
							if(lab[k] >= 1)
							{
								if(lab[k] != minlabel)
								{
									ConRegHdl->m_list[lab[k]] = minlabel;
								}
							}
						}
					}
				}
				if(lpBitData[j*iStride + i] > T && ConRegHdl->m_ptemp[j*lWidth + i]<=0)
				{
					ConRegHdl->m_ptemp[j*lWidth + i] = label;
					ConRegHdl->m_list[label++]=-1;
				}
				if(label > SAMPLE_NUMBER-3){
					bJmp	= true;
					break;
				}
			}
		}//
	}//end


	for( j=BOL; j<lHeight-BOL; j++)
	{
		for( i=BOL; i<lWidth-BOL; i++)
		{
			if(ConRegHdl->m_ptemp[j*lWidth + i] > 0)
			{
				curlab =ConRegHdl-> m_ptemp[j*lWidth + i];
				while (ConRegHdl->m_list[curlab]!=-1)
				{
					curlab = ConRegHdl->m_list[curlab];
				}
				ConRegHdl->m_ptemp[j*lWidth + i] = curlab;
				if(curlab < SAMPLE_NUMBER)
				{
					ConRegHdl->m_iCount[curlab]++;
				}
			}
		}
	}
	for(i=1; i<SAMPLE_NUMBER; i++)
	{
		if(ConRegHdl->m_iCount[i] > iscatter)
		{
			ConRegHdl->m_iRelative[max+1] = i;
			max++;
		}
	}
	patternnum = max;

	for( i=0; i<SAMPLE_NUMBER; i++)
	{
		ptn[i].lefttop.x = lWidth;
		ptn[i].lefttop.y = lHeight;
		ptn[i].rightbottom.x = 0;
		ptn[i].rightbottom.y = 0;
	}
	for(t=1; t<=patternnum; t++)
	{
		for(j=BOL; j<lHeight-BOL; j++)
		{
			for(i=BOL; i<lWidth-BOL; i++)
			{
				if(*(ConRegHdl->m_ptemp + j*lWidth + i) == (UInt8)ConRegHdl->m_iRelative[t])
				{
					if (ptn[t-1].lefttop.x > i)//get the lefttop point
						ptn[t-1].lefttop.x = i;
					if (ptn[t-1].lefttop.y > j)
						ptn[t-1].lefttop.y = j;
					if (ptn[t-1].rightbottom.x < i)//get the rightbottom point
						ptn[t-1].rightbottom.x = i;
					if (ptn[t-1].rightbottom.y < j)
						ptn[t-1].rightbottom.y = j;
				}
			}
		}
		assert(ptn[t-1].lefttop.x <= ptn[t-1].rightbottom.x);
		assert(ptn[t-1].lefttop.y <= ptn[t-1].rightbottom.y);
		assert(ptn[t-1].rightbottom.x < lWidth);
		assert(ptn[t-1].rightbottom.y < lHeight);
	}
#if 0
	m_patternnum = 0;
	memcpy(m_pPatterns, &ptn, sizeof(ptn));
	m_patternnum = patternnum;
#else
	MergeRect(ConRegHdl, ptn, patternnum);
#endif
	return iRet;
}
#else
bool    GetMoveDetect(CON_REG_HANDLE ConRegHdl, IMG_MAT src, UTC_Rect roi, int T, int iscatter)
{
	bool iRet = true;
	int lWidth, lHeight, iStride;
	UInt8 *lpBitData = (UInt8 *)src.data_u8;

	int i, j, t, k;
	int label = 1; 
	int kx, ky;
	int minlabel=-1, lab[5]; //label
	int curlab=-1;
	int max=0;

	bool	bJmp = false;
	Pattern ptn[SAMPLE_NUMBER];
	int patternnum, iCur;

	lWidth = src.width;	lHeight = src.height;	 iStride = src.step[0];
	iRet = InitializedMD(ConRegHdl, lWidth, lHeight, iStride);
	if (!iRet){
		ConRegHdl->m_patternnum = 0;
		return false;
	}

	memset(&ptn, 0, sizeof(ptn));
	memset(ConRegHdl->m_ptemp, 0, lWidth*lHeight*sizeof(UInt8));
	memset(ConRegHdl->m_iCount,0,sizeof(int)*SAMPLE_NUMBER);
	memset(ConRegHdl->m_iRelative,0,sizeof(int)*SAMPLE_NUMBER);
	memset(ConRegHdl->m_list,0,sizeof(int)*SAMPLE_NUMBER);

	memset(lpBitData, 0, iStride*roi.y);
	memset(lpBitData+iStride*(roi.y+roi.height), 0, iStride*(lHeight-roi.y-roi.height));
	for(iCur=0; iCur<lHeight; iCur++){
		memset(lpBitData+iCur*iStride, 0, roi.x);
		memset(lpBitData+iCur*iStride + roi.x+roi.width, 0, lWidth-roi.x-roi.width);
	}

	ConRegHdl->m_list[0] = -1;

	for( j=roi.y; j<(roi.y+roi.height)	&& !bJmp; j++)
	{
		for( i=roi.x; i<(roi.x+roi.width)	&& !bJmp; i++)
		{
			if(*(lpBitData + j*iStride + i) > T)
			{
				minlabel=-1;
				lab[1]=-1; lab[2]=-1; lab[3]=-1; lab[4]=-1;

				ky=j-1; kx=i+1;
				if(*(lpBitData + ky*iStride + kx)>T)
				{
					lab[1]= ConRegHdl->m_ptemp[ky*lWidth + kx];
					minlabel = ConRegHdl->m_ptemp[ky*lWidth + kx];
				}

				ky=j-1; kx=i;
				if(*(lpBitData + ky*iStride + kx)>T)
				{
					lab[2] = ConRegHdl->m_ptemp[ky*lWidth + kx];
					if(minlabel>lab[2]||minlabel==-1)
					{
						minlabel=lab[2];
					}
				}

				ky=j-1; kx=i-1;
				if(*(lpBitData + ky*iStride + kx)>T)
				{
					lab[3] = ConRegHdl->m_ptemp[ky*lWidth + kx];
					if(minlabel>lab[3]||minlabel==-1)
					{
						minlabel=lab[3];
					}
				}

				ky=j; kx=i-1;
				if(*(lpBitData + ky*iStride + kx)>T)
				{
					lab[4] = ConRegHdl->m_ptemp[ky*lWidth + kx];
					if(minlabel>lab[4]||minlabel==-1)
					{
						minlabel=lab[4];
					}
				}
				if(minlabel<0)//
				{
					ConRegHdl->m_ptemp[j*lWidth + i]=label;
					ConRegHdl->m_list[label]=-1;
					label++;
				}
				else//minlabel>=0
				{
					if(lab[1]==lab[2] && lab[2]==lab[3] && lab[3]==lab[4])
					{
						ConRegHdl->m_ptemp[j*lWidth + i] = minlabel;
					}
					else if(minlabel==(lab[1]+lab[2]+lab[3]+lab[4]+3))
					{
						ConRegHdl->m_ptemp[j*lWidth + i] = minlabel;
					}
					else
					{
						ConRegHdl->m_ptemp[j*lWidth + i] = minlabel;
						for( k=1; k<=4; k++)
						{
							if(lab[k] >= 1)
							{
								if(lab[k] != minlabel)
								{
									ConRegHdl->m_list[lab[k]] = minlabel;
								}
							}
						}
					}
				}
				if(lpBitData[j*iStride + i] > T && ConRegHdl->m_ptemp[j*lWidth + i]<=0)
				{
					ConRegHdl->m_ptemp[j*lWidth + i] = label;
					ConRegHdl->m_list[label++]=-1;
				}
				if(label > SAMPLE_NUMBER-3){
					bJmp	= true;
					break;
				}
			}
		}
	}

	for( j=roi.y; j<(roi.y+roi.height); j++)
	{
		for( i=roi.x; i<(roi.x+roi.width); i++)
		{
			if(ConRegHdl->m_ptemp[j*lWidth + i] > 0)
			{
				curlab =ConRegHdl-> m_ptemp[j*lWidth + i];
				while (ConRegHdl->m_list[curlab]!=-1)
				{
					curlab = ConRegHdl->m_list[curlab];
				}
				ConRegHdl->m_ptemp[j*lWidth + i] = curlab;
				if(curlab < SAMPLE_NUMBER)
				{
					ConRegHdl->m_iCount[curlab]++;
				}
			}
		}
	}
	for(i=1; i<SAMPLE_NUMBER; i++)
	{
		if(ConRegHdl->m_iCount[i] > iscatter)
		{
			ConRegHdl->m_iRelative[max+1] = i;
			max++;
		}
	}
	patternnum = max;

	for( i=0; i<SAMPLE_NUMBER; i++)
	{
		ptn[i].lefttop.x = lWidth-1;
		ptn[i].lefttop.y = lHeight-1;
		ptn[i].rightbottom.x = 0;
		ptn[i].rightbottom.y = 0;
	}
	for(t=1; t<=patternnum; t++)
	{
		for(j=roi.y; j<(roi.y+roi.height); j++)
		{
			for(i=roi.x; i<(roi.x+roi.width); i++)
			{
				if(*(ConRegHdl->m_ptemp + j*lWidth + i) == (UInt8)ConRegHdl->m_iRelative[t])
				{
					if (ptn[t-1].lefttop.x > i)//get the lefttop point
						ptn[t-1].lefttop.x = i;
					if (ptn[t-1].lefttop.y > j)
						ptn[t-1].lefttop.y = j;
					if (ptn[t-1].rightbottom.x < i)//get the rightbottom point
						ptn[t-1].rightbottom.x = i;
					if (ptn[t-1].rightbottom.y < j)
						ptn[t-1].rightbottom.y = j;
				}
			}
		}
		assert(ptn[t-1].lefttop.x <= ptn[t-1].rightbottom.x);
		assert(ptn[t-1].lefttop.y <= ptn[t-1].rightbottom.y);
		assert(ptn[t-1].rightbottom.x < lWidth);
		assert(ptn[t-1].rightbottom.y < lHeight);
	}
#if 0
	m_patternnum = 0;
	memcpy(m_pPatterns, &ptn, sizeof(ptn));
	m_patternnum = patternnum;
#else
	MergeRect(ConRegHdl, ptn, patternnum);
#endif
	return iRet;
}
#endif

static void	MergeRect(CON_REG_HANDLE ConRegHdl,  Pattern	ptn[], int num)
{
	int	i,	j;
	UTC_Rect rc1,	rc2, roi;
	int	status;
	for(j=0; j<num;	j++){
		ptn[j].bValid = true;
		ptn[j].bEdge = false;
		rc1.width = ptn[j].rightbottom.x -  ptn[j].lefttop.x;
		rc1.height = ptn[j].rightbottom.y -  ptn[j].lefttop.y;
		if(rc1.width < 4 || rc1.height < 4)//*******very important*******//
			ptn[j].bValid = false;
	}
	for(j=0; j<num;	j++){
		if(	!ptn[j].bValid	)
			continue;
		rc1.x = ptn[j].lefttop.x;rc1.y = ptn[j].lefttop.y;
		rc1.width = ptn[j].rightbottom.x -  ptn[j].lefttop.x;	
		rc1.height = ptn[j].rightbottom.y -  ptn[j].lefttop.y;
		for(i=j+1; i<num;	i++){
			if(	!ptn[i].bValid	)
				continue;
			rc2.x = ptn[i].lefttop.x;	rc2.y = ptn[i].lefttop.y;
			rc2.width = ptn[i].rightbottom.x -  ptn[i].lefttop.x;	
			rc2.height = ptn[i].rightbottom.y -  ptn[i].lefttop.y;
			status = _bInRect(rc1, rc2, &roi);
			if(status == 1){
				ptn[j].bValid = false;
			}else if(status == 2){
				ptn[i].bValid = false;
			}else if(status == 0){//overlap
				;
			}
		}
	}
	ConRegHdl->m_patternnum = 0;
	for(j=0; j<num;	j++){
		if( ptn[j].bValid ){
			memcpy(ConRegHdl->m_pPatterns+ConRegHdl->m_patternnum , ptn+j, sizeof(Pattern));
			ConRegHdl->m_patternnum++;
		}
	}
}
