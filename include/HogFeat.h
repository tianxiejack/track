#ifndef _HOG_FEATURE_
#define _HOG_FEATURE_

#include "float.h"

#include "DFT.h"

#define PI    3.1415926

#define EPS 0.000001

#define F_MAX FLT_MAX
#define F_MIN -FLT_MAX

		// The number of elements in bin
		// The number of sectors in gradient histogram building
#define NUM_SECTOR 9

		// The number of levels in image resize procedure
		// We need Lambda levels to resize image twice
#define LAMBDA 10

		// Block size. Used in feature pyramid building procedure
#define SIDE_LENGTH 8

#define VAL_OF_TRUNCATE 0.2f 

//modified from "_lsvm_error.h"
#define LATENT_SVM_OK 0
#define LATENT_SVM_MEM_NULL 2
#define DISTANCE_TRANSFORM_OK 1
#define DISTANCE_TRANSFORM_GET_INTERSECTION_ERROR -1
#define DISTANCE_TRANSFORM_ERROR -2
#define DISTANCE_TRANSFORM_EQUAL_POINTS -3
#define LATENT_SVM_GET_FEATURE_PYRAMID_FAILED -4
#define LATENT_SVM_SEARCH_OBJECT_FAILED -5
#define LATENT_SVM_FAILED_SUPERPOSITION -6
#define FILTER_OUT_OF_BOUNDARIES -7
#define LATENT_SVM_TBB_SCHEDULE_CREATION_FAILED -8
#define LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT -9
#define FFT_OK 2
#define FFT_ERROR -10
#define LSVM_PARSER_FILE_NOT_FOUND -11

typedef struct{
	int sizeX;
	int sizeY;
	int numFeatures;
	float *map;
} CvLSVMFeatureMapCaskade;

typedef struct _frame_buf_t{
	unsigned char *buffer[3];
	int		width;
	int		height;
	int		stride[3];
	int		channels;
}TrkFrame;

typedef struct _grad_buf_t{
	float *buffer[3];
	int		width;
	int		height;
	int		stride[3];
	int		channels;
}GradFrame;


/*
// Getting feature map for the selected subimage  
//
// API
// int getFeatureMaps(const IplImage * image, const int k, featureMap **map);
// INPUT
// image             - selected subimage
// k                 - size of cells
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/

int getFeatureMapsCR(const IMG_MAT_UCHAR * image, const int k, CvLSVMFeatureMapCaskade **map, int num_sector);

int normalizeAndTruncateCR(CvLSVMFeatureMapCaskade *map, const float alfa, int num_sector);

int PCAFeatureMapsCR(CvLSVMFeatureMapCaskade *map, int num_sector);

void GetHogFeat(const IMG_MAT_UCHAR * image, IMG_MAT_FLOAT *featMap, int cell_size, int num_sector);

#endif
