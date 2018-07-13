#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "arm_neon.h"
#include "SceneMV.h"
#include "MatMem.h"
#include "Enhance.h"

static SCENE_MV_OBJ	gSceneMVObj;
SCENE_MV_HANDLE		CreateSceneHdl()
{
	SCENE_MV_HANDLE		sceneMVHdl = (SCENE_MV_HANDLE)&gSceneMVObj;
	memset(sceneMVHdl, 0, sizeof(SCENE_MV_HANDLE));
	return sceneMVHdl;
}

void CloseSceneHdl(SCENE_MV_HANDLE pSceneMVObj)
{
	if(pSceneMVObj != NULL){
		_unInitSceneMV(pSceneMVObj);
	}
}

void _initSceneMV(SCENE_MV_OBJ* pSceneMVObj, IMG_MAT *image)
{
	int i;

	if(pSceneMVObj->m_width != image->width || pSceneMVObj->m_height != image->height){
		_unInitSceneMV(pSceneMVObj);
		pSceneMVObj->m_width = image->width;
		pSceneMVObj->m_height = image->height;
	}

	pSceneMVObj->m_sceneMV.num = 0;/***********/
	pSceneMVObj->m_curOptConf.opt = 0.f;
	pSceneMVObj->m_curOptConf.pos.x = pSceneMVObj->m_curOptConf.pos.y = 0;
	pSceneMVObj->m_curMVMode = CUR_UN_KNOW;

	InitTrajectory(pSceneMVObj);

	int width, height, centW, centH;
	UTC_Rect	scenePos;
	width = image->width;
	height = image->height;
	if(width<=768){ //720x576
		scenePos.width = 72;
		scenePos.height = 60;
	}else if(width<=1024){//1024x768
		scenePos.width = 100;
		scenePos.height = 72;
	}else if(width<=1280){//1280x1024
		scenePos.width = 120;
		scenePos.height = 90;
	}else{//1920x1080
		scenePos.width = 156;
		scenePos.height = 96;
	}
	scenePos.x =scenePos.width*1.75;
	scenePos.y = scenePos.height*1.75;
	scenePos.x &=(~1);
	scenePos.y &=(~1);
	for(i=0; i<MAX_SCENE_BLOCK; i++){
		int startx = ((i%2)==0)?(width/2-scenePos.x-scenePos.width):(width/2+scenePos.x);
		int starty = ((i/2)==0)?(height/2-scenePos.y-scenePos.height):(height/2+scenePos.y);
		pSceneMVObj->m_sceneParam.refPos[i].x = (startx&(~1));
		pSceneMVObj->m_sceneParam.refPos[i].y = (starty&(~1));
		pSceneMVObj->m_sceneParam.refPos[i].width = scenePos.width;
		pSceneMVObj->m_sceneParam.refPos[i].height = scenePos.height;
//		printf("Ref[%d] Pos Rect(%d, %d, %d, %d) \n", i, startx, starty, scenePos.width, scenePos.height);

		pSceneMVObj->m_sceneParam.curPos[i].x = pSceneMVObj->m_sceneParam.refPos[i].x - scenePos.width*0.25;
		pSceneMVObj->m_sceneParam.curPos[i].y = pSceneMVObj->m_sceneParam.refPos[i].y - scenePos.height*0.25;
		pSceneMVObj->m_sceneParam.curPos[i].width = scenePos.width*1.5;
		pSceneMVObj->m_sceneParam.curPos[i].height = scenePos.height*1.5;
		pSceneMVObj->m_sceneParam.curPos[i].x &= (~1);
		pSceneMVObj->m_sceneParam.curPos[i].y &= (~1);
		pSceneMVObj->m_sceneParam.curPos[i].width &= (~1);
		pSceneMVObj->m_sceneParam.curPos[i].height &= (~1);
//		printf("Cur[%d] Pos Rect(%d, %d, %d, %d) \n", i, pSceneMVObj->m_sceneParam.curPos[i].x, pSceneMVObj->m_sceneParam.curPos[i].y, pSceneMVObj->m_sceneParam.curPos[i].width, pSceneMVObj->m_sceneParam.curPos[i].height);
	}

	SCENE_PARAM *pScenePar = (SCENE_PARAM*)&pSceneMVObj->m_sceneParam;
	for(i=0; i<MAX_SCENE_BLOCK; i++){
		pSceneMVObj->_refSceneMap[i] = matAlloc(image->dtype,	pScenePar->refPos[i].width, pScenePar->refPos[i].height, image->channels);
		pSceneMVObj->_curSceneMap[i] = matAlloc(image->dtype,	pScenePar->curPos[i].width, pScenePar->curPos[i].height, image->channels);
	}
}

void _unInitSceneMV(SCENE_MV_OBJ* pSceneMVObj)
{
	int i;
	for(i=0; i<MAX_SCENE_BLOCK; i++){
		matFree(pSceneMVObj->_refSceneMap[i]);
		pSceneMVObj->_refSceneMap[i] = NULL;
		matFree(pSceneMVObj->_curSceneMap[i]);
		pSceneMVObj->_curSceneMap[i] = NULL;
	}
	clrTrkState(pSceneMVObj);
	memset(pSceneMVObj->m_curPosConf, 0, sizeof(SCENE_POS_CONF)*MAX_SCENE_BLOCK);
}

static float _getVar(IMG_MAT *ref)
{
	int i,size;
	float meanv = 0.f, vsquared = 0.f, var;
	size = ref->width*ref->height;
#pragma UNROLL(4)
	for(i=0; i<size; i++){
		meanv += ref->data_u8[i];
		vsquared += ref->data_u8[i]*ref->data_u8[i];
	}
	meanv /=  size; /* mean */
	vsquared /= size; /* mean (x^2) */
	var = ( vsquared - (meanv * meanv) );//
	var = (float)sqrt(var); /* var */
	return var;
}

static void _getScene(SCENE_MV_OBJ* pSceneMVObj, IMG_MAT *image, int type)
{
	int i;
	SCENE_PARAM *pScenePar = (SCENE_PARAM *)&pSceneMVObj->m_sceneParam;
	Recti window[MAX_SCENE_BLOCK];

	if(type ==REF_IMAGE_TYPE){
#pragma omp parallel for
		for(i=0; i<MAX_SCENE_BLOCK; i++){
			window[i].x = pScenePar->refPos[i].x;					window[i].y = pScenePar->refPos[i].y;
			window[i].width = pScenePar->refPos[i].width;	window[i].height = pScenePar->refPos[i].height;
			CutWindowMat(*image, pSceneMVObj->_refSceneMap[i], window[i]);
			pSceneMVObj->refVar[i] = _getVar(pSceneMVObj->_refSceneMap[i]);
		}
	}else {//==CUR_IMAGE_TYPE
#pragma omp parallel for
		for(i=0; i<MAX_SCENE_BLOCK; i++){
			window[i].x = pScenePar->curPos[i].x;					window[i].y = pScenePar->curPos[i].y;
			window[i].width = pScenePar->curPos[i].width;	window[i].height = pScenePar->curPos[i].height;
			CutWindowMat(*image, pSceneMVObj->_curSceneMap[i], window[i]);
		}
	}
}

static void _getSceneEnh(SCENE_MV_OBJ* pSceneMVObj, IMG_MAT *image, int type, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit)
{
	int i;
	SCENE_PARAM *pScenePar = (SCENE_PARAM *)&pSceneMVObj->m_sceneParam;
	Recti window[MAX_SCENE_BLOCK];

	if(type ==REF_IMAGE_TYPE){
#pragma omp parallel for
		for(i=0; i<MAX_SCENE_BLOCK; i++){

			window[i].x = pScenePar->refPos[i].x-pScenePar->curPos[i].x;					window[i].y = pScenePar->refPos[i].y-pScenePar->curPos[i].y;
			window[i].width = pScenePar->refPos[i].width;												window[i].height = pScenePar->refPos[i].height;
			CutWindowMat(*pSceneMVObj->_curSceneMap[i], pSceneMVObj->_refSceneMap[i], window[i]);
			pSceneMVObj->refVar[i] = _getVar(pSceneMVObj->_refSceneMap[i]);
		}
	}else {//==CUR_IMAGE_TYPE
#pragma omp parallel for
		for(i=0; i<MAX_SCENE_BLOCK; i++){
			window[i].x = pScenePar->curPos[i].x;					window[i].y = pScenePar->curPos[i].y;
			window[i].width = pScenePar->curPos[i].width;	window[i].height = pScenePar->curPos[i].height;
			CutWindowMat(*image, pSceneMVObj->_curSceneMap[i], window[i]);
			CLAHE_enh_U8((kz_pixel_t*)pSceneMVObj->_curSceneMap[i]->data_u8, pSceneMVObj->_curSceneMap[i]->width, pSceneMVObj->_curSceneMap[i]->height,
										 0, 0, pSceneMVObj->_curSceneMap[i]->width, pSceneMVObj->_curSceneMap[i]->height, 0, 255, uiNrX, uiNrY, 256, fCliplimit);
		}
	}
}

#if 0
static void	compSceneSimilar(SCENE_MV_OBJ* pSceneMVObj,  IMG_MAT *image, IMG_MAT *cur, IMG_MAT *ref, SCENE_POS_CONF *posConf, int posID)
{
	float fNcc = 0.f;

	IMG_MAT_UCHAR	*_img = (IMG_MAT_UCHAR*)cur;
	IMG_MAT_UCHAR	*_tmpl = (IMG_MAT_UCHAR *)ref;
	IMG_MAT	*result = matAlloc(MAT_float, _img->width-_tmpl->width+1, _img->height-_tmpl->height+1, 1);
	IMG_MAT_FLOAT	*_result = (IMG_MAT_FLOAT	*)result;
	SCENE_PARAM *pScenePar = (SCENE_PARAM*)&pSceneMVObj->m_sceneParam;
	posConf->pos.x = 0;
	posConf->pos.y = 0;

	matchTemplateUchar(*_img, *_tmpl, *_result, CR_TM_CCOEFF_NORMED);

	if(result->width == 1 && result->height == 1){
		fNcc = _result->data[0];
	}else{
		int i, j;
		float maxNcc = 0.f;
		float	*prslt;

		for(j=0; j<result->height; j++){
			prslt = result->data + result->step[0]*j;
			for(i=0; i<result->width; i++){
				if(i==0 && j==0)
					continue;
				if(prslt[i] > maxNcc ){
					maxNcc = prslt[i];
					posConf->pos.x = i;
					posConf->pos.y = j;
				}
			}
		}
		fNcc = maxNcc;
	}
	pScenePar->trkPos[posID].x = pScenePar->curPos[posID].x + posConf->pos.x;
	pScenePar->trkPos[posID].y = pScenePar->curPos[posID].y + posConf->pos.y;
	pScenePar->trkPos[posID].width = pScenePar->refPos[posID].width;
	pScenePar->trkPos[posID].height = pScenePar->refPos[posID].height;

	posConf->opt = fNcc;
	posConf->pos.x = posConf->pos.x + pScenePar->curPos[posID].x - pScenePar->refPos[posID].x;
	posConf->pos.y = posConf->pos.y + pScenePar->curPos[posID].y  - pScenePar->refPos[posID].y;

	matFree(_result);
}
#else
static void	compSceneSimilar(SCENE_MV_OBJ* pSceneMVObj,  IMG_MAT *image, IMG_MAT *cur, IMG_MAT *ref, SCENE_POS_CONF *posConf, int posID)
{
	SCENE_PARAM *pScenePar = (SCENE_PARAM*)&pSceneMVObj->m_sceneParam;
	cv::Mat	_tmplModel, _curImg, result;
	double minVal, maxVal;
	 cv::Point minLoc,maxLoc;
	_curImg = cv::Mat(cur->height, cur->width, CV_8UC1, cur->data_u8);
	_tmplModel = cv::Mat(ref->height, ref->width, CV_8UC1, ref->data_u8);
	posConf->pos.x = 0;
	posConf->pos.y = 0;

	cv::matchTemplate(_curImg, _tmplModel, result, CV_TM_CCOEFF_NORMED);
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	posConf->pos.x = maxLoc.x;
	posConf->pos.y = maxLoc.y;
	pScenePar->trkPos[posID].x = pScenePar->curPos[posID].x + posConf->pos.x;
	pScenePar->trkPos[posID].y = pScenePar->curPos[posID].y + posConf->pos.y;
	pScenePar->trkPos[posID].width = pScenePar->refPos[posID].width;
	pScenePar->trkPos[posID].height = pScenePar->refPos[posID].height;

	if(maxLoc.x == 0 && maxLoc.y == 0)
		maxVal = 0.0;
	posConf->opt = (float)maxVal;
	posConf->pos.x = posConf->pos.x + pScenePar->curPos[posID].x - pScenePar->refPos[posID].x;
	posConf->pos.y = posConf->pos.y + pScenePar->curPos[posID].y  - pScenePar->refPos[posID].y;

}
#endif

static float calEuclidean(SCENE_POS_CONF conf1, SCENE_POS_CONF conf2)
{
	return (float)(sqrt((conf1.pos.x-conf2.pos.x)*(conf1.pos.x-conf2.pos.x)+(conf1.pos.y-conf2.pos.y)*(conf1.pos.y-conf2.pos.y)));
}

#define GET_AVE_OPT_POS_3(optConf, I, J, K)	\
		optConf.opt = ((pSceneMVObj->m_curPosConf[I].opt+pSceneMVObj->m_curPosConf[J].opt+pSceneMVObj->m_curPosConf[K].opt)/3.0);		\
		optConf.pos.x = floor((pSceneMVObj->m_curPosConf[I].pos.x+pSceneMVObj->m_curPosConf[J].pos.x+pSceneMVObj->m_curPosConf[K].pos.x)/3.0+0.5);	\
		optConf.pos.y = floor((pSceneMVObj->m_curPosConf[I].pos.y+pSceneMVObj->m_curPosConf[J].pos.y+pSceneMVObj->m_curPosConf[K].pos.y)/3.0+0.5);

#define GET_AVE_OPT_POS_2(optConf, I, J)	\
		optConf.opt = (pSceneMVObj->m_curPosConf[I].opt+pSceneMVObj->m_curPosConf[J].opt)/2;		\
		optConf.pos.x = floor((pSceneMVObj->m_curPosConf[I].pos.x+pSceneMVObj->m_curPosConf[J].pos.x)/2.0+0.5);	\
		optConf.pos.y = floor((pSceneMVObj->m_curPosConf[I].pos.y+pSceneMVObj->m_curPosConf[J].pos.y)/2.0+0.5);

#define REGET_OPT_POS(optConf, I, J, maxIdx)	\
		if(pSceneMVObj->refVar[I]<15 && pSceneMVObj->refVar[J]<15 && pSceneMVObj->refVar[maxIdx] > 25.0){	\
			optConf.opt = max_Conf.opt;	\
			optConf.pos = max_Conf.pos;	\
		}else if(pSceneMVObj->refVar[I]<15 && pSceneMVObj->refVar[J]<15 && pSceneMVObj->refVar[maxIdx] < 15){	\
			bConf = false;	\
		}

static void calSceneSimilar(SCENE_MV_OBJ* pSceneMVObj, IMG_MAT *image)
{
	int i, j;
	SCENE_POS_CONF	max_Conf, aveConf, optConf;
	PointfCR		sumPt, optPt;
	int maxIdx = 0;
	float  euLen01, euLen02, euLen03,euLen12, euLen13, euLen23;
	float  euLen[MAX_SCENE_BLOCK][MAX_SCENE_BLOCK];
	float  euThred = 2.0;
	bool	bConf = true;
	float   optThred = 0.75;
	memset(euLen, 0, sizeof(euLen));

#pragma omp parallel for
	for(i=0; i<MAX_SCENE_BLOCK; i++){
		compSceneSimilar(pSceneMVObj, image, pSceneMVObj->_curSceneMap[i], pSceneMVObj->_refSceneMap[i], &pSceneMVObj->m_curPosConf[i], i);
	}

	aveConf.pos.x = aveConf.pos.y = 0;
	aveConf.opt = 0.f;
	max_Conf.opt = 0.f;
	for(i=0; i<MAX_SCENE_BLOCK; i++){
		aveConf.opt +=pSceneMVObj->m_curPosConf[i].opt;
		aveConf.pos.x += pSceneMVObj->m_curPosConf[i].pos.x;
		aveConf.pos.y += pSceneMVObj->m_curPosConf[i].pos.y;
		if(pSceneMVObj->m_curPosConf[i].opt > max_Conf.opt){
			max_Conf.opt = pSceneMVObj->m_curPosConf[i].opt;
			max_Conf.pos = pSceneMVObj->m_curPosConf[i].pos;
			maxIdx = i;
		}
	}

	for(j=0; j<MAX_SCENE_BLOCK; j++){
		for(i= j+1; i<MAX_SCENE_BLOCK; i++){
			euLen[j][i] = calEuclidean(pSceneMVObj->m_curPosConf[i], pSceneMVObj->m_curPosConf[j]);
			euLen[i][j] = euLen[j][i];
		}
	}

#if 0
	if((fabs(pSceneMVObj->m_curPosConf[0].pos.y)>6 &&pSceneMVObj->m_curPosConf[0].opt>optThred) ||
		(fabs(pSceneMVObj->m_curPosConf[1].pos.y)>6 &&pSceneMVObj->m_curPosConf[1].opt>optThred)||
		(fabs(pSceneMVObj->m_curPosConf[2].pos.y)>6 &&pSceneMVObj->m_curPosConf[2].opt>optThred) ||
		(fabs(pSceneMVObj->m_curPosConf[3].pos.y)>6  &&pSceneMVObj->m_curPosConf[3].opt>optThred) ){
		for(i=0; i<4; i++){
			printf("%s:pos[%d]=(%d, %d) opt=%0.2f\n",__func__, i, pSceneMVObj->m_curPosConf[i].pos.x, pSceneMVObj->m_curPosConf[i].pos.y, pSceneMVObj->m_curPosConf[i].opt);
		}
	}
#endif
	if( ((euLen[0][1]<euThred && euLen[0][2]<euThred) || (euLen[0][1]<euThred && euLen[1][2]<euThred)|| (euLen[0][2]<euThred && euLen[1][2]<euThred) )&&
			pSceneMVObj->m_curPosConf[0].opt > optThred  && pSceneMVObj->m_curPosConf[1].opt > optThred && pSceneMVObj->m_curPosConf[2].opt > optThred){
		GET_AVE_OPT_POS_3(optConf, 0, 1, 2);
	}else if( ((euLen[0][1]<euThred && euLen[0][3]<euThred) || (euLen[0][1]<euThred && euLen[1][3]<euThred)|| (euLen[0][2]<euThred && euLen[1][3]<euThred)) &&
			pSceneMVObj->m_curPosConf[0].opt > optThred  && pSceneMVObj->m_curPosConf[1].opt > optThred && pSceneMVObj->m_curPosConf[3].opt > optThred){
		GET_AVE_OPT_POS_3(optConf, 0, 1, 3);
	}else if( ((euLen[0][2]<euThred && euLen[0][3]<euThred) || (euLen[0][2]<euThred && euLen[2][3]<euThred)|| (euLen[0][3]<euThred && euLen[2][3]<euThred)) &&
			pSceneMVObj->m_curPosConf[0].opt > optThred  && pSceneMVObj->m_curPosConf[2].opt > optThred && pSceneMVObj->m_curPosConf[3].opt > optThred){
		GET_AVE_OPT_POS_3(optConf, 0, 2, 3);
	}else if( ((euLen[1][2]<euThred && euLen[1][3]<euThred)  || (euLen[1][2]<euThred && euLen[2][3]<euThred)|| (euLen[2][3]<euThred && euLen[1][3]<euThred)) &&
			pSceneMVObj->m_curPosConf[1].opt > optThred  && pSceneMVObj->m_curPosConf[2].opt > optThred && pSceneMVObj->m_curPosConf[3].opt > optThred){
		GET_AVE_OPT_POS_3(optConf, 1, 2, 3);
	}else if(euLen[0][1]<euThred && pSceneMVObj->m_curPosConf[0].opt > optThred  && pSceneMVObj->m_curPosConf[1].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 0, 1);
		REGET_OPT_POS(optConf, 0, 1, maxIdx);
	}else if(euLen[0][2]<euThred && pSceneMVObj->m_curPosConf[0].opt > optThred  && pSceneMVObj->m_curPosConf[2].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 0, 2);
		REGET_OPT_POS(optConf, 0, 2, maxIdx);
	}else if(euLen[0][3]<euThred && pSceneMVObj->m_curPosConf[0].opt > optThred  && pSceneMVObj->m_curPosConf[3].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 0, 3);
		REGET_OPT_POS(optConf, 0, 3, maxIdx);
	}else if(euLen[1][2]<euThred && pSceneMVObj->m_curPosConf[1].opt > optThred  && pSceneMVObj->m_curPosConf[2].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 1, 2);
		REGET_OPT_POS(optConf, 1, 2, maxIdx);
	}else if(euLen[1][3]<euThred && pSceneMVObj->m_curPosConf[1].opt > optThred  && pSceneMVObj->m_curPosConf[3].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 1, 3);
		REGET_OPT_POS(optConf, 1, 3, maxIdx);
	}else if(euLen[2][3]<euThred && pSceneMVObj->m_curPosConf[2].opt > optThred  && pSceneMVObj->m_curPosConf[3].opt > optThred){
		GET_AVE_OPT_POS_2(optConf, 2, 3);
		REGET_OPT_POS(optConf, 2, 3, maxIdx);
	}else{
		optConf.opt = max_Conf.opt;
		optConf.pos = max_Conf.pos;
		if(pSceneMVObj->refVar[maxIdx] < 30.0)
			bConf = false;
	}

	pSceneMVObj->_bOptConf = bConf;

	sumPt.x = sumPt.y = 0;
	if( pSceneMVObj->m_sceneMV.num > 7 && bConf ){//fast reverse direction is possible,but this state is error
		for(i=0; i<7; i++){
			sumPt.x += pSceneMVObj->m_sceneMV.mv[pSceneMVObj->m_sceneMV.num-1-i].x;
			sumPt.y += pSceneMVObj->m_sceneMV.mv[pSceneMVObj->m_sceneMV.num-1-i].y;
		}
		optPt.x = optConf.pos.x;			optPt.y = optConf.pos.y;
		sumPt.x /= 7;								sumPt.y /= 7;
		float dif_dist = sqrt((sumPt.x-optPt.x)*(sumPt.x-optPt.x) + (sumPt.y-optPt.y)*(sumPt.y-optPt.y));
		if(dif_dist > 10.0 && ( (fabs(sumPt.x-optPt.x)>6 && sumPt.x*optPt.x<0) ||( fabs(sumPt.y-optPt.y)>6 && sumPt.y*optPt.y<0) )){//direction is opposite
			pSceneMVObj->_bOptConf = false;
		}
	}

	pSceneMVObj->m_curOptConf = optConf;

	if(!pSceneMVObj->_bOptConf)
		return;

	if(pSceneMVObj->m_sceneMV.num<TRACK_POS_NUM){
		pSceneMVObj->m_sceneMV.mv[pSceneMVObj->m_sceneMV.num]= optConf.pos;
		pSceneMVObj->m_sceneMV.num++;
	}else{
		memmove((void*)pSceneMVObj->m_sceneMV.mv, (void*)(pSceneMVObj->m_sceneMV.mv+1), (pSceneMVObj->m_sceneMV.num-1)*sizeof(PointfCR));
		memmove((void*)pSceneMVObj->m_sceneMV.conf, (void*)(pSceneMVObj->m_sceneMV.conf+1), (pSceneMVObj->m_sceneMV.num-1)*sizeof(float));
		pSceneMVObj->m_sceneMV.mv[pSceneMVObj->m_sceneMV.num-1]= optConf.pos;
		pSceneMVObj->m_sceneMV.conf[pSceneMVObj->m_sceneMV.num-1]= optConf.opt;
	}
	pSceneMVObj->m_curMVMode = CUR_UN_KNOW;
	if(pSceneMVObj->m_sceneMV.num >= 10){
		PointICR *pt = pSceneMVObj->m_sceneMV.mv+pSceneMVObj->m_sceneMV.num-10;
		int stillCount = 0;
		for(i=0; i<10; i++){
			if(pt[i].x==0 && pt[i].y == 0)
				stillCount++;
		}
		if(stillCount >= 6)
			pSceneMVObj->m_curMVMode = CUR_STATIC;
		else
			pSceneMVObj->m_curMVMode = CUR_MOVE;
	}
}

int getSceneMap(SCENE_MV_OBJ* pSceneMVObj, IMG_MAT *image,  bool init)
{
	if(pSceneMVObj->m_width != image->width || pSceneMVObj->m_height != image->height)
		init = true;
	if(init){
//		_unInitSceneMV(pDftTrkObj);
		_initSceneMV(pSceneMVObj, image);
		_getScene(pSceneMVObj, image, REF_IMAGE_TYPE);
		return 0;
	}
	_getScene(pSceneMVObj, image, CUR_IMAGE_TYPE);
	calSceneSimilar(pSceneMVObj, image);
	_getScene(pSceneMVObj, image, REF_IMAGE_TYPE);
	return 0;
}

int getSceneMapEnh(SCENE_MV_OBJ* pSceneMVObj, IMG_MAT *image,  bool init, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit)
{
	if(pSceneMVObj->m_width != image->width || pSceneMVObj->m_height != image->height)
		init = true;
	if(init){

		_initSceneMV(pSceneMVObj, image);
		_getSceneEnh(pSceneMVObj, image, CUR_IMAGE_TYPE, uiNrX, uiNrY, fCliplimit);
		_getSceneEnh(pSceneMVObj, image, REF_IMAGE_TYPE, uiNrX, uiNrY, fCliplimit);
		return 0;
	}
	_getSceneEnh(pSceneMVObj, image, CUR_IMAGE_TYPE, uiNrX, uiNrY, fCliplimit);
	calSceneSimilar(pSceneMVObj, image);
	_getSceneEnh(pSceneMVObj, image, REF_IMAGE_TYPE, uiNrX, uiNrY, fCliplimit);
	return 0;
}



