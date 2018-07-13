#include "PCTracker.h"
#include "DFT.h"


void gaussianCorrelation(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, IMG_MAT_FLOAT dst, int size_patch[], bool _hogfeatures, float sigma)
{

	int i;
	IMG_MAT_FLOAT c1, c2,c3, d, midMat;
	UTILS_assert(src0.channels == 1 && src1.channels == 1 && dst.channels == 1);
	UTILS_assert(src0.width == src1.width && src0.height == src1.height);
	UTILS_assert(dst.width == size_patch[1] && dst.height == size_patch[0]);
	AllocIMGMat(&c1, size_patch[1], size_patch[0], 1);
	AllocIMGMat(&d, size_patch[1], size_patch[0], 1);
	AllocIMGMat(&midMat, size_patch[1], size_patch[0], 2);
	
	// HOG features
	if (_hogfeatures) {
		IMG_MAT_FLOAT caux,	x1aux,	x2aux;
		AllocIMGMat(&caux, size_patch[1], size_patch[0], 2);
		AllocIMGMat(&x1aux, size_patch[1], size_patch[0], 2);
		AllocIMGMat(&x2aux, size_patch[1], size_patch[0], 2);
		AllocIMGMat(&c3, size_patch[1], size_patch[0], 1);

		 for (i = 0; i < size_patch[2]; i++) {
			 float *psrc = src0.data + i*src0.step[0];
			 MergeFloat2Cplex(psrc, size_patch[1], size_patch[0], &x1aux);
			 psrc = src1.data + i*src1.step[0];
			 MergeFloat2Cplex(psrc,size_patch[1], size_patch[0], &x2aux);
			 dftCR(x1aux, x1aux, 0, 0);
			 dftCR(x2aux, x2aux, 0, 0);

			 mulSpectrumsCR(x1aux, x2aux, caux, 0, true);
			 idftCR(caux, caux, CR_DFT_INVERSE | CR_DFT_SCALE, 0);
			 rearrangeCR(&caux, midMat);
			 realCR(caux, &c3);
			 AddMat(c1, c3, c1);
		 }
		 
		 FreeIMGMat(&c3);
		 FreeIMGMat(&x2aux);
		 FreeIMGMat(&x1aux);
		 FreeIMGMat(&caux);

	}else{// Gray features
		IMG_MAT_FLOAT s0, s1;
		AllocIMGMat(&s0, src0.width, src0.height, 2);
		AllocIMGMat(&s1, src1.width, src1.height, 2);
		AllocIMGMat(&c2, size_patch[1], size_patch[0], 2);

		MergeFloat2Cplex(src0.data, src0.width, src0.height, &s0);
		MergeFloat2Cplex(src1.data, src1.width, src1.height, &s1);

		dftCR(s0, s0, 0, 0);
		dftCR(s1, s1, 0, 0);
		mulSpectrumsCR(s0, s1, c2, 0, true);
		idftCR(c2, c2, CR_DFT_INVERSE | CR_DFT_SCALE, 0);
		rearrangeCR(&c2, midMat);
		realCR(c2, &c1);

		FreeIMGMat(&c2);
		FreeIMGMat(&s1);
		FreeIMGMat(&s0);
	}
	maxCmp(src0, src1, c1, d, size_patch);
	ExpMat(d, dst, sigma);

	FreeIMGMat(&midMat);
	FreeIMGMat(&d);
	FreeIMGMat(&c1);
}

void maxCmp(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1,  IMG_MAT_FLOAT src2, IMG_MAT_FLOAT dst, int size_patch[])
{
	int k;
	int nWidth = src0.width;
	int nHeight = src0.height;
	int nchannels = src0.channels;
	float *psrc0, *psrc1, *psrc2, *pdst;
	float sum0 =0.0, sum1 = 0.0, sum, multi, reslt;

	UTILS_assert(src0.channels == 1 && src1.channels == 1 && src2.channels == 1 && dst.channels == 1);

	psrc0 = src0.data;
	psrc1 = src1.data;
	for(k=0; k<nWidth*nHeight; k++)
	{
		sum0 += psrc0[k]*psrc0[k];
		sum1 += psrc1[k]*psrc1[k];
	}
	sum = sum0 + sum1;
	multi = 1.f/(size_patch[0]*size_patch[1]*size_patch[2]);
	nWidth = src2.width;
	nHeight = src2.height;
	psrc2 = src2.data;
	pdst = dst.data;
	#pragma UNROLL(4)
	for(k=0; k<nWidth*nHeight; k++)
	{
		reslt = (sum-2*psrc2[k])*multi ;
		pdst[k] = (reslt>0)?reslt:0.0;
//		if( reslt>0){
//			pdst[k] = reslt;
//		}else{
//			pdst[k] = 0.0;
//		}
	}
}

void createGaussianPeak(IMG_MAT_FLOAT dst, int sizey, int sizex, float padding, float output_sigma_factor)
{
	int syh = (sizey) / 2;
	int sxh = (sizex) / 2;
	int i, j;

	float output_sigma = sqrt((float) sizex * sizey) / padding * output_sigma_factor;
	float mult = -0.5 / (output_sigma * output_sigma);

	IMG_MAT_FLOAT midMat;
	float *pmid;
	AllocIMGMat(&midMat, dst.width, dst.height, 1);
	for (i = 0; i < sizey; i++)
	{
		for (j = 0; j < sizex; j++)
		{
			int ih = i - syh;
			int jh = j - sxh;
			pmid = midMat.data + i*sizex + j;
			*pmid = exp(mult * (float) (ih * ih + jh * jh));
		}
	}
	MergeFloat2Cplex(midMat.data, midMat.width, midMat.height, &dst);
	FreeIMGMat(&midMat);
	dftCR(dst, dst, 0, 0);
}

void trainCR(IMG_MAT_FLOAT src, IMG_MAT_FLOAT _prob, IMG_MAT_FLOAT *_tmpl, IMG_MAT_FLOAT *_alphaf, float train_interp_factor, float lambda,
					int size_patch[], bool _hogfeatures, float sigma)
{
	//unsigned int *p;
	//float *pf;
	IMG_MAT_FLOAT k, K, alphaf;
	AllocIMGMat(&k, size_patch[1], size_patch[0], 1);
	AllocIMGMat(&K, size_patch[1], size_patch[0], 2);
	AllocIMGMat(&alphaf, _prob.width, _prob.height, _prob.channels);

	gaussianCorrelation(src, src, k, size_patch, _hogfeatures, sigma);
	MergeFloat2Cplex(k.data, k.width, k.height, &K);
	dftCR(K, K, 0, 0);
	AddK(K, lambda, K);
	complexDivisionCR(_prob, K, alphaf);

	MultiplyCR(*_tmpl, src, train_interp_factor, *_tmpl);
	MultiplyCR(*_alphaf, alphaf, train_interp_factor, *_alphaf);


#if 0			
	p = (unsigned int *)_tmpl->data;
	pf = (float*)p + (_tmpl->width*_tmpl->height*_tmpl->channels/4);
	
	Vps_printf("dft _tmpl: %.6f %.6f %.6f %.6f\n",
		*pf, *(pf+1), *(pf+2), *(pf+3));
	Vps_printf("dft _tmpl: %.6f %.6f %.6f %.6f\n",
		*(pf+4), *(pf+5), *(pf+6), *(pf+7));
	Vps_printf("dft _tmpl: %.6f %.6f %.6f %.6f\n",
		*(pf+8), *(pf+9), *(pf+10), *(pf+11));
	Vps_printf("dft _tmpl: %.6f %.6f %.6f %.6f\n",
		*(pf+12), *(pf+13), *(pf+14), *(pf+15));

	p = (unsigned int *)_alphaf->data;
	pf = (float*)p + (_alphaf->width*_alphaf->height*_alphaf->channels/4);
	
	Vps_printf("dft _alphaf: %.6f %.6f %.6f %.6f\n",
		*pf, *(pf+1), *(pf+2), *(pf+3));
	Vps_printf("dft _alphaf: %.6f %.6f %.6f %.6f\n",
		*(pf+4), *(pf+5), *(pf+6), *(pf+7));
	Vps_printf("dft _alphaf: %.6f %.6f %.6f %.6f\n",
		*(pf+8), *(pf+9), *(pf+10), *(pf+11));
	Vps_printf("dft _alphaf: %.6f %.6f %.6f %.6f\n",
		*(pf+12), *(pf+13), *(pf+14), *(pf+15));

#endif	

	FreeIMGMat(&alphaf);
	FreeIMGMat(&K);
	FreeIMGMat(&k);
}

void MultiplyCR(IMG_MAT_FLOAT src0, IMG_MAT_FLOAT src1, float K, IMG_MAT_FLOAT dst)
{
	int nWidth = src0.width;
	int nHeight = src0.height;
	int nChannel = src0.channels;
	int k;
	float *psrc0, *psrc1, *pdst;
	ComplexCR *psrclex0, *psrclex1, *pdstlex;
	UTILS_assert(src0.width == src1.width && src0.height == src1.height);
	UTILS_assert(src1.width == dst.width && src1.height == dst.height);
	UTILS_assert(src0.channels == dst.channels && src1.channels == dst.channels );

	if(nChannel == 1){
		psrc0 = (float *)(src0.data);
		psrc1 = (float *)(src1.data);
		pdst = (float *)(dst.data);
		#pragma UNROLL(4)
		for(k=0; k<nWidth*nHeight; k++)
		{
			pdst[k] = (1 - K)*psrc0[k] + K*psrc1[k]; 
		}
	}else if(nChannel == 2){
		psrclex0 = (ComplexCR *)(src0.data);
		psrclex1 = (ComplexCR *)(src1.data);
		pdstlex = (ComplexCR *)(dst.data);
		#pragma UNROLL(4)
		for(k=0; k<nWidth*nHeight; k++)
		{
			pdstlex[k].re = (1 - K)*psrclex0[k].re + K*psrclex1[k].re; 
			pdstlex[k].im = (1 - K)*psrclex0[k].im + K*psrclex1[k].im; 
		}
	}
}

void detectCR(IMG_MAT_FLOAT z, IMG_MAT_FLOAT x, IMG_MAT_FLOAT *res,  IMG_MAT_FLOAT _alphaf, int size_patch[], bool _hogfeatures, float sigma)
{
	IMG_MAT_FLOAT k, K;

	//Vps_printf("detectCR enter !!!!!!!!!\n");
	UTILS_assert(z.width == x.width && z.height == x.height);
	UTILS_assert(z.channels == 1 && x.channels == 1 && res->channels == 1);

#if 0			
	unsigned int *p = (unsigned int *)z.data;
	float *pf = (float*)p + (z.width*z.height*z.channels/4);
	//*pf = 0.1f; *(pf+1)=0.1f; *(pf+2)=0.1f; *(pf+3)=0.1f;
	
	Vps_printf("dft z: %.6f %.6f %.6f %.6f\n",
		*pf, *(pf+1), *(pf+2), *(pf+3));
	Vps_printf("dft z: %.6f %.6f %.6f %.6f\n",
		*(pf+4), *(pf+5), *(pf+6), *(pf+7));
	Vps_printf("dft z: %.6f %.6f %.6f %.6f\n",
		*(pf+8), *(pf+9), *(pf+10), *(pf+11));
	Vps_printf("dft z: %.6f %.6f %.6f %.6f\n",
		*(pf+12), *(pf+13), *(pf+14), *(pf+15));
	Vps_printf("dft z: (%d,%d,%d),hog:%d,sigma:%f \n",
		size_patch[0],size_patch[1], size_patch[2],
		_hogfeatures, sigma);
#endif	
	
	AllocIMGMat(&k, size_patch[1], size_patch[0], 1);
	AllocIMGMat(&K, size_patch[1], size_patch[0], 2);

	gaussianCorrelation(x, z, k, size_patch, _hogfeatures, sigma);
	MergeFloat2Cplex(k.data, k.width, k.height, &K);
#if 0			
	p = (unsigned int *)K.data;
	pf = (float*)p + (K.width*K.height*K.channels/4);
	//*pf = 0.1f; *(pf+1)=0.1f; *(pf+2)=0.1f; *(pf+3)=0.1f;
	
	Vps_printf("dft K: %.6f %.6f %.6f %.6f\n",
		*pf, *(pf+1), *(pf+2), *(pf+3));
	Vps_printf("dft K: %.6f %.6f %.6f %.6f\n",
		*(pf+4), *(pf+5), *(pf+6), *(pf+7));
	Vps_printf("dft K: %.6f %.6f %.6f %.6f\n",
		*(pf+8), *(pf+9), *(pf+10), *(pf+11));
	Vps_printf("dft K: %.6f %.6f %.6f %.6f\n",
		*(pf+12), *(pf+13), *(pf+14), *(pf+15));
#endif	
	
	dftCR(K,K,0,0);
	complexMultiCR(_alphaf, K, K);
	idftCR(K, K, CR_DFT_INVERSE | CR_DFT_SCALE, 0);
	realCR(K, res);

	FreeIMGMat(&K);
	FreeIMGMat(&k);
}

void getMinMaxValue(IMG_MAT_FLOAT res, PointfCR *_maxfIdx, float *_peak_value)
{
	float MaxValue, peak_value;
	PointICR MaxIdx;
	PointfCR	MaxfIdx;

	MinMaxLoc(res, NULL, &MaxValue, NULL, &MaxIdx);
	 peak_value = (float) MaxValue;
	 MaxfIdx.x = (float)MaxIdx.x;
	 MaxfIdx.y = (float)MaxIdx.y;

	 if (MaxIdx.x > 0 && MaxIdx.x < res.width-1) {
		 float *psr0 = res.data + MaxIdx.y*res.step[0]+(MaxIdx.x-1)*res.channels;
		 float *psr1= res.data + MaxIdx.y*res.step[0]+(MaxIdx.x+1)*res.channels;
		 MaxfIdx.x += SubPixelPeak( *psr0 , peak_value,  *psr1);
	 }

	 if (MaxIdx.y > 0 && MaxIdx.y < res.height-1) {
		 float *psr0 = res.data + (MaxIdx.y-1)*res.step[0]+MaxIdx.x*res.channels;
		 float *psr1 = res.data + (MaxIdx.y+1)*res.step[0]+MaxIdx.x*res.channels;
		 MaxfIdx.y += SubPixelPeak(*psr0, peak_value, *psr1);
	 }

	 MaxfIdx.x -= (res.width) / 2;//fft transform after, shift image
	 MaxfIdx.y -= (res.height) / 2;

	 if(_maxfIdx)
		 *_maxfIdx = MaxfIdx;
	 if(_peak_value != NULL)
		 *_peak_value = peak_value;
}
