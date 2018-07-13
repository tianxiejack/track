#include "PCTracker.h"
#include "DFT.h"
#include	"HogFeat.h"
#include "neon_std.h"

//static float *_mempace_hog = NULL;

static int allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj, const int sizeX, const int sizeY, const int numFeatures)
{
	(*obj) = (CvLSVMFeatureMapCaskade *)AllocSpaceCR(sizeof(CvLSVMFeatureMapCaskade));//malloc(sizeof(CvLSVMFeatureMapCaskade));
	(*obj)->sizeX       = sizeX;
	(*obj)->sizeY       = sizeY;
	(*obj)->numFeatures = numFeatures;

	(*obj)->map = (float *)AllocSpaceCR(sizeof (float) * (sizeX * sizeY  * numFeatures));//malloc(sizeof (float) * (sizeX * sizeY  * numFeatures));
	//_mempace_hog += (sizeX * sizeY  * numFeatures);

	memset((*obj)->map, 0, sizeX * sizeY * numFeatures*sizeof(float));
	
	return LATENT_SVM_OK;
}

static int freeFeatureMapObject (CvLSVMFeatureMapCaskade **obj)
{
	if(*obj == NULL) return LATENT_SVM_MEM_NULL;
	
	FreeSpaceCR((*obj)->map);
	
	FreeSpaceCR(*obj);
	(*obj) = NULL;
	
	return LATENT_SVM_OK;
}

#if 0
static void convFilter2D(const IMG_MAT_UCHAR * image, IMG_MAT_FLOAT *grad, int method /*= 0*/)//method==>0:Horiz; 1:Verti
{
	int i,	j;
	int nWidth = image->width;
	int	nHeight = image->height;
	int	nChannels = image->channels;
	int	step = image->step[0];
	unsigned char *psrc;
	unsigned char *psrc1;
	float	*pdst;

	UTILS_assert(image != NULL && grad != NULL && image->channels == 1);
	UTILS_assert(image->width == grad->width && image->height == grad->height && image->channels == grad->channels);

	if(method == 0)
	{
		for(j=0; j<nHeight; j++)
		{
			psrc = image->data_u8+j*image->step[0];
			pdst = grad->data+j*grad->step[0];
			#pragma UNROLL(4)
			for(i=1; i<nWidth-1; i++)
			{
				pdst[i] = (float)psrc[i+1]-(float)psrc[i-1];
			}
		}
	}
	else
	{
		for(j=1; j<nHeight-1; j++)
		{
			psrc = image->data_u8+(j-1)*image->step[0];
			psrc1 = image->data_u8+(j+1)*image->step[0];
			pdst = grad->data+j*grad->step[0];
			#pragma UNROLL(4)
			for(i=0; i<nWidth; i++)
			{
				pdst[i] = (float)psrc1[i]-(float)psrc[i];
			}
		}
	}
}
#else
static void convFilter2D(const IMG_MAT_UCHAR * image, IMG_MAT_FLOAT *grad, int method /*= 0*/)//method==>0:Horiz; 1:Verti
{
	int i,	j;
	int nWidth = image->width;
	int	nHeight = image->height;
	int	nChannels = image->channels;
	int	step = image->step[0];

	uint8x8_t * __restrict__ pSrc08x8_t, * __restrict__ pSrc18x8_t, * __restrict__ pDst8x8_t;

	unsigned char *psrc;
	unsigned char *psrc1;
	float	*pdst;

	UTILS_assert(image != NULL && grad != NULL && image->channels == 1);
	UTILS_assert(image->width == grad->width && image->height == grad->height && image->channels == grad->channels);

	if(method == 0)
	{
		for(j=0; j<nHeight; j++)
		{
			psrc = image->data_u8+j*image->step[0];
			pdst = grad->data+j*grad->step[0];
			//#pragma UNROLL(4)
			//for(i=1; i<nWidth-1; i++)
			//{
			//	pdst[i] = ((float)psrc[i+1]-(float)psrc[i-1]);
			//}
			sub_u8_float_neon(pdst, &psrc[2], &psrc[0], nWidth-2);
		}
	}
	else
	{
		for(j=1; j<nHeight-1; j++)
		{
			psrc = image->data_u8+(j-1)*image->step[0];
			psrc1 = image->data_u8+(j+1)*image->step[0];
			pdst = grad->data+j*grad->step[0];
			//#pragma UNROLL(4)
			//for(i=0; i<nWidth; i++)
			//{
			//	pdst[i] = ((float)psrc1[i]-(float)psrc[i]);
			//}
			sub_u8_float_neon(pdst, psrc1, psrc, nWidth);
		}
	}
}
#endif

#if 0
int getFeatureMapsCR(const IMG_MAT_UCHAR * image, const int k, CvLSVMFeatureMapCaskade **map, int num_sector)
{
	UTILS_assert(image->channels == 1);
	int sizeX, sizeY;
	int p, px, stringSize;
	int height, width, numChannels;
	int i, j, kk, c, ii, jj, d;
	float  * datadx, * datady;

	int   ch; 
	float magnitude, x, y, tx, ty;

	IMG_MAT_FLOAT *dx, *dy;
	IMG_MAT_FLOAT gradX, gradY;
	int *nearest;
	float *w, a_x, b_x;

	float kernel[3] = {-1.f, 0.f, 1.f};

	float * r;
	int   * alfa;

	float boundary_x[128];//[NUM_SECTOR + 1];
	float boundary_y[128];//[NUM_SECTOR + 1];
	float max, dotProd;
	int   maxi;

	//float *_mempace_hogBak = _mempace_hog;

	height = image->height;
	width  = image->width ;

	numChannels = image->channels;

	AllocIMGMat(&gradX, image->width, image->height, numChannels);
	AllocIMGMat(&gradY, image->width, image->height, numChannels);
	dx = &gradX;
	dy = &gradY;

	//dx->data = _mempace_hog;
	//_mempace_hog += image->width*image->height;
	//dy->data = _mempace_hog;
	//_mempace_hog += image->width*image->height;

	sizeX = width  / k;
	sizeY = height / k;
	px    = 3 * num_sector; 
	p     = px;
	stringSize = sizeX * p;
	allocFeatureMapObject(map, sizeX, sizeY, p);

	convFilter2D(image, dx, 0);
	convFilter2D(image, dy, 1);

	float arg_vector;
	for(i = 0; i <= num_sector; i++)
	{
		arg_vector    = ( (float) i ) * ( (float)(PI) / (float)(num_sector) );
		boundary_x[i] = cosf(arg_vector);
		boundary_y[i] = sinf(arg_vector);
	}/*for(i = 0; i <= NUM_SECTOR; i++) */

	r    = (float *)AllocSpaceCR( sizeof(float) * (width * height));//malloc( sizeof(float) * (width * height));
	alfa = (int   *)AllocSpaceCR( sizeof(int  ) * (width * height * 2));//malloc( sizeof(int  ) * (width * height * 2));

	for(j = 1; j < height - 1; j++)
	{
		datadx = (float*)(dx->data + dx->step[0] * j);
		datady = (float*)(dy->data + dy->step[0] * j);
		for(i = 1; i < width - 1; i++)
		{
			c = 0;
			x = (datadx[i * numChannels + c]);
			y = (datady[i * numChannels + c]);

			r[j * width + i] =sqrtf(x * x + y * y);
			for(ch = 1; ch < numChannels; ch++)
			{
				tx = (datadx[i * numChannels + ch]);
				ty = (datady[i * numChannels + ch]);
				magnitude = sqrtf(tx * tx + ty * ty);
				if(magnitude > r[j * width + i])
				{
					r[j * width + i] = magnitude;
					c = ch;
					x = tx;
					y = ty;
				}
			}/*for(ch = 1; ch < numChannels; ch++)*/

			max  = boundary_x[0] * x + boundary_y[0] * y;
			maxi = 0;
			for (kk = 0; kk < num_sector; kk++) 
			{
				dotProd = boundary_x[kk] * x + boundary_y[kk] * y;
				if (dotProd > max) 
				{
					max  = dotProd;
					maxi = kk;
				}
				else 
				{
					if (-dotProd > max) 
					{
						max  = -dotProd;
						maxi = kk + num_sector;
					}
				}
			}
			alfa[j * width * 2 + i * 2    ] = maxi % num_sector;
			alfa[j * width * 2 + i * 2 + 1] = maxi;  
		}/*for(i = 0; i < width; i++)*/
	}/*for(j = 0; j < height; j++)*/

	//_mempace_hog = _mempace_hogBak;

	nearest = (int  *)AllocSpaceCR(sizeof(int  ) *  k);//malloc(sizeof(int  ) *  k);
	w       = (float*)AllocSpaceCR(sizeof(float) * (k * 2));//malloc(sizeof(float) * (k * 2));

	for(i = 0; i < k / 2; i++)
	{
		nearest[i] = -1;
	}/*for(i = 0; i < k / 2; i++)*/
	for(i = k / 2; i < k; i++)
	{
		nearest[i] = 1;
	}/*for(i = k / 2; i < k; i++)*/

	for(j = 0; j < k / 2; j++)
	{
		b_x = k / 2 + j + 0.5f;
		a_x = k / 2 - j - 0.5f;
		w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x)); 
		w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));  
	}/*for(j = 0; j < k / 2; j++)*/
	for(j = k / 2; j < k; j++)
	{
		a_x = j - k / 2 + 0.5f;
		b_x =-j + k / 2 - 0.5f + k;
		w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x)); 
		w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));  
	}/*for(j = k / 2; j < k; j++)*/

	for(i = 0; i < sizeY; i++)
	{
		for(j = 0; j < sizeX; j++)
		{
			for(ii = 0; ii < k; ii++)
			{
				for(jj = 0; jj < k; jj++)
				{
					if ((i * k + ii > 0) && 
						(i * k + ii < height - 1) && 
						(j * k + jj > 0) && 
						(j * k + jj < width  - 1))
					{
						d = (k * i + ii) * width + (j * k + jj);
						(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]] += 
							r[d] * w[ii * 2] * w[jj * 2];
						(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += 
							r[d] * w[ii * 2] * w[jj * 2];
						if ((i + nearest[ii] >= 0) && 
							(i + nearest[ii] <= sizeY - 1))
						{
							(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]             ] += 
								r[d] * w[ii * 2 + 1] * w[jj * 2 ];
							(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += 
								r[d] * w[ii * 2 + 1] * w[jj * 2 ];
						}
						if ((j + nearest[jj] >= 0) && 
							(j + nearest[jj] <= sizeX - 1))
						{
							(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]             ] += 
								r[d] * w[ii * 2] * w[jj * 2 + 1];
							(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += 
								r[d] * w[ii * 2] * w[jj * 2 + 1];
						}
						if ((i + nearest[ii] >= 0) && 
							(i + nearest[ii] <= sizeY - 1) && 
							(j + nearest[jj] >= 0) && 
							(j + nearest[jj] <= sizeX - 1))
						{
							(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]             ] += 
								r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
							(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += 
								r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
						}
					}
				}/*for(jj = 0; jj < k; jj++)*/
			}/*for(ii = 0; ii < k; ii++)*/
		}/*for(j = 1; j < sizeX - 1; j++)*/
	}/*for(i = 1; i < sizeY - 1; i++)*/

	FreeSpaceCR(w);
	FreeSpaceCR(nearest);

	FreeSpaceCR(r);
	FreeSpaceCR(alfa);

	FreeIMGMat(&gradX);
	FreeIMGMat(&gradY);

	return LATENT_SVM_OK;
}

#else

int getFeatureMapsCR(const IMG_MAT_UCHAR * image, const int k, CvLSVMFeatureMapCaskade **map, int num_sector)
{
	int sizeX, sizeY;
	int p, px, stringSize;
	int height, width, numChannels;
	int i, j, kk, c, ii, jj, d,dt;
	float  * datadx, * datady;
	float x, y;

	IMG_MAT_FLOAT *dx, *dy;
	IMG_MAT_FLOAT gradX, gradY;

	float a_x, b_x;

	float kernel[3] = {-1.f, 0.f, 1.f};

	float * r;
	int   * alfa;

	float max, dotProd;
	int   maxi;

	float ftmp00, ftmp01, ftmp10, ftmp11;
	static float boundary_x[128];//[NUM_SECTOR + 1];
	static float boundary_y[128];//[NUM_SECTOR + 1];
	static int nearest[64];
	static float w[32];
	static float ww00[32][32];
	static float ww01[32][32];
	static float ww10[32][32];
	static float ww11[32][32];

	{
		static bool bInit = false;
		if(!bInit)
		{
			float arg_vector;
			for(i = 0; i <= num_sector; i++)
			{
				arg_vector    = ( (float) i ) * ( (float)(PI) / (float)(num_sector) );
				boundary_x[i] = cosf(arg_vector);
				boundary_y[i] = sinf(arg_vector);
			}/*for(i = 0; i <= NUM_SECTOR; i++) */

			float a_x, b_x;

			for(i = 0; i < k / 2; i++)
			{
				nearest[i] = -1;
			}/*for(i = 0; i < k / 2; i++)*/
			for(i = k / 2; i < k; i++)
			{
				nearest[i] = 1;
			}/*for(i = k / 2; i < k; i++)*/

			for(j = 0; j < k / 2; j++)
			{
				b_x = k / 2 + j + 0.5f;
				a_x = k / 2 - j - 0.5f;
				w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x));
				w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));
			}/*for(j = 0; j < k / 2; j++)*/
			for(j = k / 2; j < k; j++)
			{
				a_x = j - k / 2 + 0.5f;
				b_x =-j + k / 2 - 0.5f + k;
				w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x));
				w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));
			}/*for(j = k / 2; j < k; j++)*/

			for(j=0; j<k; j++){
				for(i=0; i<k; i++){
					ww00[j][i] = w[j*2]*w[i*2];
					ww01[j][i] = w[j*2]*w[i*2+1];
					ww10[j][i] = w[j*2+1]*w[i*2];
					ww11[j][i] = w[j*2+1]*w[i*2+1];
				}
			}

			bInit = true;
		}
	}

	assert(image->channels == 1);

	height = image->height;
	width  = image->width ;

	numChannels = image->channels;

	AllocIMGMat(&gradX, image->width, image->height, numChannels);
	AllocIMGMat(&gradY, image->width, image->height, numChannels);

	dx = &gradX;
	dy = &gradY;

	sizeX = width  / k;
	sizeY = height / k;
	px    = 3 * num_sector; 
	p     = px;
	stringSize = sizeX * p;
	allocFeatureMapObject(map, sizeX, sizeY, p);

	convFilter2D(image, dx, 0);
	convFilter2D(image, dy, 1);

	r    = (float *)AllocSpaceCR( sizeof(float) * (width * height));//malloc( sizeof(float) * (width * height));
	alfa = (int   *)AllocSpaceCR( sizeof(int  ) * (width * height * 2));//malloc( sizeof(int  ) * (width * height * 2));

	for(j = 1; j < height - 1; j++)
	{
		datadx = (float*)(dx->data + dx->step[0] * j);
		datady = (float*)(dy->data + dy->step[0] * j);
		for(i = 1; i < width - 1; i++)
		{
			c = 0;
			x = datadx[i];//(datadx[i * numChannels + c]);
			y = datady[i ];//(datady[i * numChannels + c]);

			r[j * width + i] =sqrtf(x * x + y * y);
			
			max  = boundary_x[0] * x + boundary_y[0] * y;
			maxi = 0;

			if(num_sector == 9)
			{
				float32x4_t tmpx = vmulq_n_f32(vld1q_f32(&boundary_x[1]), x);
				float32x4_t tmpy = vmulq_n_f32(vld1q_f32(&boundary_y[1]), y);
				float32x4_t v_41 = vaddq_f32(tmpx, tmpy);
				tmpx = vmulq_n_f32(vld1q_f32(&boundary_x[5]), x);
				tmpy = vmulq_n_f32(vld1q_f32(&boundary_y[5]), y);
				float32x4_t v_42 = vaddq_f32(tmpx, tmpy);

				dotProd = max;
				if (dotProd > max)
				{
					max  = dotProd;
					maxi = 0;
				}
				else
				{
					if (-dotProd > max)
					{
						max  = -dotProd;
						maxi = 0 + num_sector;
					}
				}

				for (kk = 0; kk < 4; kk++)
				{
					dotProd = vgetq_lane_f32(v_41, kk);
					if (dotProd > max)
					{
						max  = dotProd;
						maxi = kk+1;
					}
					else
					{
						if (-dotProd > max)
						{
							max  = -dotProd;
							maxi = kk+1 + num_sector;
						}
					}
				}
				for (kk = 0; kk < 4; kk++)
				{
					dotProd = vgetq_lane_f32(v_42, kk);
					if (dotProd > max)
					{
						max  = dotProd;
						maxi = kk+5;
					}
					else
					{
						if (-dotProd > max)
						{
							max  = -dotProd;
							maxi = kk+5 + num_sector;
						}
					}
				}
				alfa[j * width * 2 + i * 2    ] = maxi % num_sector;
				alfa[j * width * 2 + i * 2 + 1] = maxi;
			}
			else
			{
				for (kk = 0; kk < num_sector; kk++)
				{
					dotProd = boundary_x[kk] * x + boundary_y[kk] * y;
					if (dotProd > max)
					{
						max  = dotProd;
						maxi = kk;
					}
					else
					{
						if (-dotProd > max)
						{
							max  = -dotProd;
							maxi = kk + num_sector;
						}
					}
				}
			}
			alfa[j * width * 2 + i * 2    ] = maxi % num_sector;
			alfa[j * width * 2 + i * 2 + 1] = maxi;  
		}/*for(i = 0; i < width; i++)*/
	}/*for(j = 0; j < height; j++)*/

	//mid part
	for(i = 1; i < sizeY-1; i++)
	{
		for(j = 1; j < sizeX-1; j++)	
		{
			for(ii=0; ii<k; ii++)
			{
				dt = (k * i + ii) * width + (j * k + 0);
				#pragma UNROLL(4)
				for(jj=0; jj<k; jj++)
				{
					d = dt + jj;
					ftmp00 = r[d] * ww00[ii][jj];
					ftmp10 = r[d] * ww10[ii][jj];
					ftmp01 = r[d] * ww01[ii][jj];
					ftmp11 = r[d] * ww11[ii][jj];

					(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]] +=  ftmp00;
					(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] +=  ftmp00;

					(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp10;
					(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp10;

					(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp01;
					(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp01;

					(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]	] += ftmp11;
					(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp11;
				}
			}
		}
	}

	//Vps_printf("%s 3: %d us\n", __func__, Utils_getCurTimeInUsec()-tm);
	//left border
	for(i = 1; i < sizeY-1; i++)
	{
		j = 0;
		for(ii = 0; ii < k; ii++)
		{
			for(jj = k/2; jj < k; jj++)
			{
				d = (k * i + ii) * width + (j * k + jj);
				ftmp00 = r[d] * ww00[ii][jj];
				ftmp10 = r[d] * ww10[ii][jj];
				ftmp01 = r[d] * ww01[ii][jj];
				ftmp11 = r[d] * ww11[ii][jj];

				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]] +=  ftmp00;
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] +=  ftmp00;

				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp10;
				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp10;

				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp01;
				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp01;

				(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]	] += ftmp11;
				(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp11;
				
			}/*for(jj = 0; jj < k; jj++)*/
			{
				jj = 1;
				d = (k * i + ii) * width + (j * k + jj);
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]] += r[d] * ww00[ii][jj];//w[ii * 2] * w[jj * 2];
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += r[d] * ww00[ii][jj];//w[ii * 2] * w[jj * 2];

				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]		] += r[d] * ww10[ii][jj];//w[ii * 2 + 1] * w[jj * 2 ];
				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += r[d] * ww10[ii][jj];//w[ii * 2 + 1] * w[jj * 2 ];

			}/*for(jj = 0; jj < k; jj++)*/
		}/*for(ii = 0; ii < k; ii++)*/
	}
	//Vps_printf("%s 4: %d us\n", __func__, Utils_getCurTimeInUsec()-tm);

	//right border
	for(i = 1; i < sizeY-1; i++)
	{
		j = sizeX-1;
		for(ii = 0; ii < k; ii++)
		{
			for(jj = 0; jj < k/2; jj++)
			{
				d = (k * i + ii) * width + (j * k + jj);
				ftmp00 = r[d] * ww00[ii][jj];
				ftmp10 = r[d] * ww10[ii][jj];
				ftmp01 = r[d] * ww01[ii][jj];
				ftmp11 = r[d] * ww11[ii][jj];

				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]] +=  ftmp00;
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] +=  ftmp00;

				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp10;
				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp10;

				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp01;
				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp01;

				(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]	] += ftmp11;
				(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp11;

			}/*for(jj = 0; jj < k; jj++)*/
			{
				jj = k/2;
				d = (k * i + ii) * width + (j * k + jj);
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]] += r[d] * ww00[ii][jj];//w[ii * 2] * w[jj * 2];
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += r[d] * ww00[ii][jj];//w[ii * 2] * w[jj * 2];

				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]		] += r[d] * ww10[ii][jj];//w[ii * 2 + 1] * w[jj * 2 ];
				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += r[d] * ww10[ii][jj];//w[ii * 2 + 1] * w[jj * 2 ];

			}/*for(jj = 0; jj < k; jj++)*/
		}/*for(ii = 0; ii < k; ii++)*/
	}
	//Vps_printf("%s 5: %d us\n", __func__, Utils_getCurTimeInUsec()-tm);

	//above border
	for(j = 1; j < sizeX-1; j++)
	{
		i = 0;
		for(ii = k/2; ii < k; ii++)
		{
			#pragma UNROLL(4)
			for(jj = 0; jj < k; jj++)
			{
				d = (k * i + ii) * width + (j * k + jj);
				ftmp00 = r[d] * ww00[ii][jj];
				ftmp10 = r[d] * ww10[ii][jj];
				ftmp01 = r[d] * ww01[ii][jj];
				ftmp11 = r[d] * ww11[ii][jj];

				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]] +=  ftmp00;
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] +=  ftmp00;

				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp10;
				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp10;

				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp01;
				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp01;

				(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]	] += ftmp11;
				(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp11;

			}/*for(jj = 0; jj < k; jj++)*/
		}/*for(ii = 0; ii < k; ii++)*/
		{
			ii = 1;
#pragma UNROLL(4)
			for(jj = 0; jj < k; jj++)
			{
				d = (k * i + ii) * width + (j * k + jj);
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]] += r[d] * ww00[ii][jj];//w[ii * 2] * w[jj * 2];
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += r[d] * ww00[ii][jj];//w[ii * 2] * w[jj * 2];

				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]		] += r[d] * ww01[ii][jj];//w[ii * 2] * w[jj * 2 + 1];
				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += r[d] * ww01[ii][jj];//w[ii * 2] * w[jj * 2 + 1];
			}
		}/*for(jj = 0; jj < k; jj++)*/
	}
	//Vps_printf("%s 6: %d us\n", __func__, Utils_getCurTimeInUsec()-tm);

	//below border
	for(j = 1; j < sizeX-1; j++)
	{
		i = sizeY-1;
		for(ii = 0; ii < k/2; ii++)
		{
#pragma UNROLL(4)
			for(jj = 0; jj < k; jj++)
			{
				d = (k * i + ii) * width + (j * k + jj);
				ftmp00 = r[d] * ww00[ii][jj];
				ftmp10 = r[d] * ww10[ii][jj];
				ftmp01 = r[d] * ww01[ii][jj];
				ftmp11 = r[d] * ww11[ii][jj];

				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]] +=  ftmp00;
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] +=  ftmp00;

				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp10;
				(*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp10;

				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]		] += ftmp01;
				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp01;

				(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1)    ]	] += ftmp11;
				(*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[(d<<1) + 1] + num_sector] += ftmp11;

			}/*for(jj = 0; jj < k; jj++)*/
		}/*for(ii = 0; ii < k; ii++)*/
		{
			ii = k/2;
#pragma UNROLL(4)
			for(jj = 0; jj < k; jj++)
			{
				d = (k * i + ii) * width + (j * k + jj);
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]] += r[d] * ww00[ii][jj];//w[ii * 2] * w[jj * 2];
				(*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += r[d] * ww00[ii][jj];//w[ii * 2] * w[jj * 2];

				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]		] += r[d] * ww01[ii][jj];//w[ii * 2] * w[jj * 2 + 1];
				(*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + num_sector] += r[d] * ww01[ii][jj];//w[ii * 2] * w[jj * 2 + 1];
			}
		}/*for(jj = 0; jj < k; jj++)*/
	}

	FreeSpaceCR(r);
	FreeSpaceCR(alfa);

	FreeIMGMat(&gradX);
	FreeIMGMat(&gradY);

	return LATENT_SVM_OK;
}

#endif
/*
// Feature map Normalization and Truncation 
//
// API
// int normalizeAndTruncate(featureMap *map, const float alfa);
// INPUT
// map               - feature map
// alfa              - truncation threshold
// OUTPUT
// map               - truncated and normalized feature map
// RESULT
// Error status
*/
#define VOPT (1)
int normalizeAndTruncateCR(CvLSVMFeatureMapCaskade *map, const float alfa, int num_sector)
{
	int i,j, ii;
	int sizeX, sizeY, p, pos, pp, xp, pos1, pos2;
	float * partOfNorm; // norm of C(i, j)
	float * newData;
	float   valOfNorm;
	float    valOfNorm0,valOfNorm1,valOfNorm2,valOfNorm3;

	sizeX     = map->sizeX;
	sizeY     = map->sizeY;
	partOfNorm = (float *)AllocSpaceCR(sizeof(float) * (sizeX * sizeY));//malloc (sizeof(float) * (sizeX * sizeY));

	p  = num_sector;
	xp = num_sector * 3;
	pp = num_sector * 12;

	for(i = 0; i < sizeX * sizeY; i++)
	{
		valOfNorm = 0.0f;
		pos = i * map->numFeatures;
		for(j = 0; j < p; j++)
		{
			valOfNorm += map->map[pos + j] * map->map[pos + j];
		}/*for(j = 0; j < p; j++)*/
		partOfNorm[i] = valOfNorm;
	}/*for(i = 0; i < sizeX * sizeY; i++)*/

	sizeX -= 2;
	sizeY -= 2;

	newData = (float *)AllocSpaceCR(sizeof(float) * (sizeX * sizeY * pp));//malloc (sizeof(float) * (sizeX * sizeY * pp));
	//_mempace_hog += (sizeX * sizeY * pp);
	//normalization

	for(i = 1; i <= sizeY; i++)
	{
		for(j = 1; j <= sizeX; j++)
		{
			valOfNorm0 = 1.0f/(sqrtf(
				partOfNorm[(i    )*(sizeX + 2) + (j    )] +
				partOfNorm[(i    )*(sizeX + 2) + (j + 1)] +
				partOfNorm[(i + 1)*(sizeX + 2) + (j    )] +
				partOfNorm[(i + 1)*(sizeX + 2) + (j + 1)]) + FLT_EPSILON);
			valOfNorm1 = 1.0f/(sqrtf(
				partOfNorm[(i    )*(sizeX + 2) + (j    )] +
				partOfNorm[(i    )*(sizeX + 2) + (j + 1)] +
				partOfNorm[(i - 1)*(sizeX + 2) + (j    )] +
				partOfNorm[(i - 1)*(sizeX + 2) + (j + 1)]) + FLT_EPSILON);
			valOfNorm2 = 1.0f/(sqrtf(
				partOfNorm[(i    )*(sizeX + 2) + (j    )] +
				partOfNorm[(i    )*(sizeX + 2) + (j - 1)] +
				partOfNorm[(i + 1)*(sizeX + 2) + (j    )] +
				partOfNorm[(i + 1)*(sizeX + 2) + (j - 1)]) + FLT_EPSILON);
			valOfNorm3 = 1.0f/(sqrtf(
				partOfNorm[(i    )*(sizeX + 2) + (j    )] +
				partOfNorm[(i    )*(sizeX + 2) + (j - 1)] +
				partOfNorm[(i - 1)*(sizeX + 2) + (j    )] +
				partOfNorm[(i - 1)*(sizeX + 2) + (j - 1)]) + FLT_EPSILON);
			pos1 = (i  ) * (sizeX + 2) * xp + (j  ) * xp;
			pos2 = (i-1) * (sizeX    ) * pp + (j-1) * pp;

			if(0)
			{
				#pragma UNROLL(4)
				for(ii = 0; ii < p; ii++)
				{
					newData[pos2 + ii        ] = map->map[pos1 + ii    ] * valOfNorm0;
					newData[pos2 + ii + p    ] = map->map[pos1 + ii    ] * valOfNorm1;
					newData[pos2 + ii + p * 2] = map->map[pos1 + ii    ] * valOfNorm2;
					newData[pos2 + ii + p * 3 ] = map->map[pos1 + ii    ] * valOfNorm3;
				}/*for(ii = 0; ii < p; ii++)*/

				#pragma UNROLL(4)
				for(ii = 0; ii < 2 * p; ii++)
				{
					newData[pos2 + ii + p * 4] = map->map[pos1 + ii + p] * valOfNorm0;
					newData[pos2 + ii + p * 6] = map->map[pos1 + ii + p] * valOfNorm1;
					newData[pos2 + ii + p * 8] = map->map[pos1 + ii + p] * valOfNorm2;
					newData[pos2 + ii + p * 10] = map->map[pos1 + ii + p] * valOfNorm3;
				}/*for(ii = 0; ii < 2 * p; ii++)*/
			}
			else
			{
				mul_n_float_neon(&newData[pos2], &map->map[pos1], valOfNorm0, p);
				mul_n_float_neon(&newData[pos2 + p], &map->map[pos1], valOfNorm1, p);
				mul_n_float_neon(&newData[pos2 + p * 2], &map->map[pos1], valOfNorm2, p);
				mul_n_float_neon(&newData[pos2 + p * 3], &map->map[pos1], valOfNorm3, p);

				mul_n_float_neon(&newData[pos2 + p * 4], &map->map[pos1 + p], valOfNorm0, p*2);
				mul_n_float_neon(&newData[pos2 + p * 6], &map->map[pos1 + p], valOfNorm1, p*2);
				mul_n_float_neon(&newData[pos2 + p * 8], &map->map[pos1 + p], valOfNorm2, p*2);
				mul_n_float_neon(&newData[pos2 + p * 10], &map->map[pos1 + p], valOfNorm3, p*2);
			}
		}/*for(j = 1; j <= sizeX; j++)*/
	}/*for(i = 1; i <= sizeY; i++)*/
	//truncation
#pragma UNROLL(4)
	for(i = 0; i < sizeX * sizeY * pp; i++)
	{
//		if(newData [i] > alfa) newData [i] = alfa;
		newData [i] = (newData [i] > alfa)?alfa:newData [i];
	}/*for(i = 0; i < sizeX * sizeY * pp; i++)*/
	//swop data

	map->numFeatures  = pp;
	map->sizeX = sizeX;
	map->sizeY = sizeY;

	FreeSpaceCR (map->map);
	FreeSpaceCR (partOfNorm);

	map->map = newData;

	return LATENT_SVM_OK;
}
/*
// Feature map reduction
// In each cell we reduce dimension of the feature vector
// according to original paper special procedure
//
// API
// int PCAFeatureMaps(featureMap *map)
// INPUT
// map               - feature map
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int PCAFeatureMapsCR(CvLSVMFeatureMapCaskade *map, int num_sector)
{ 
	int i,j, ii, jj, k;
	int sizeX, sizeY, p,  pp, xp, yp, pos1, pos2;
	float * newData;
	float val;
	float nx, ny;

	sizeX = map->sizeX;
	sizeY = map->sizeY;
	p     = map->numFeatures;
	pp    = num_sector * 3 + 4;
	yp    = 4;
	xp    = num_sector;

	nx    = 1.0f / sqrtf((float)(xp * 2));
	ny    = 1.0f / sqrtf((float)(yp    ));

	newData = (float *)AllocSpaceCR(sizeof(float) * (sizeX * sizeY * pp));//malloc (sizeof(float) * (sizeX * sizeY * pp));
	//_mempace_hog += (sizeX * sizeY * pp);

	for(i = 0; i < sizeY; i++)
	{
		for(j = 0; j < sizeX; j++)
		{
			pos1 = ((i)*sizeX + j)*p;
			pos2 = ((i)*sizeX + j)*pp;
			k = 0;
			for(jj = 0; jj < xp * 2; jj++)
			{
				val = 0;
				for(ii = 0; ii < yp; ii++)
				{
					val += map->map[pos1 + yp * xp + ii * xp * 2 + jj];
				}/*for(ii = 0; ii < yp; ii++)*/
				newData[pos2 + k] = val * ny;
				k++;
			}/*for(jj = 0; jj < xp * 2; jj++)*/
			for(jj = 0; jj < xp; jj++)
			{
				val = 0;
				for(ii = 0; ii < yp; ii++)
				{
					val += map->map[pos1 + ii * xp + jj];
				}/*for(ii = 0; ii < yp; ii++)*/
				newData[pos2 + k] = val * ny;
				k++;
			}/*for(jj = 0; jj < xp; jj++)*/
			for(ii = 0; ii < yp; ii++)
			{
				val = 0;
				for(jj = 0; jj < 2 * xp; jj++)
				{
					val += map->map[pos1 + yp * xp + ii * xp * 2 + jj];
				}/*for(jj = 0; jj < xp; jj++)*/
				newData[pos2 + k] = val * nx;
				k++;
			} /*for(ii = 0; ii < yp; ii++)*/           
		}/*for(j = 0; j < sizeX; j++)*/
	}/*for(i = 0; i < sizeY; i++)*/
	//swop data

	map->numFeatures = pp;

	FreeSpaceCR (map->map);

	map->map = newData;

	return LATENT_SVM_OK;
}

#if 0
void GetHogFeat(const IMG_MAT_UCHAR * image, IMG_MAT_FLOAT *featMap, int cell_size, int num_sector)
{
	assert(featMap != NULL && featMap->channels == 1);
	
	CvLSVMFeatureMapCaskade *map;
	
	getFeatureMapsCR(image, cell_size, &map, num_sector);
	normalizeAndTruncateCR(map, 0.2f, num_sector);
	PCAFeatureMapsCR(map, num_sector);

	assert(featMap->height == (map->sizeX*map->sizeY) && featMap->width == map->numFeatures );
	memcpy(featMap->data, map->map, featMap->width*featMap->height*sizeof(float));
}
#else
void GetHogFeat(const IMG_MAT * image, IMG_MAT *featMap, int cell_size, int num_sector)
{
	//UInt32 tm = Utils_getCurTimeInMsec();

	//_mempace_hog = _mempace;
	
	CvLSVMFeatureMapCaskade *map;
	
	UTILS_assert(featMap != NULL && featMap->channels == 1);

	getFeatureMapsCR(image, cell_size, &map, num_sector);
	//Vps_printf("FET0: %d ms\n", Utils_getCurTimeInMsec()-tm);
	normalizeAndTruncateCR(map, 0.2f, num_sector);
	//Vps_printf("FET1: %d ms\n", Utils_getCurTimeInMsec()-tm);
	PCAFeatureMapsCR(map, num_sector);
	//Vps_printf("FET2: %d ms\n", Utils_getCurTimeInMsec()-tm);

	UTILS_assert(featMap->height == (map->sizeX*map->sizeY) && featMap->width == map->numFeatures );
	memcpy(featMap->data, map->map, featMap->width*featMap->height*sizeof(float));

	freeFeatureMapObject(&map);
}
#endif

