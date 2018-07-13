#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "SaliencyProc.h"

using namespace cv;
using namespace std;

cv::Mat GetSR(cv::Mat image, cv::Size sz)
{
//	int64 t_sr = getTickCount();
	cv::Size resizedImageSize = sz;
	 std::vector<cv::Mat> mv;
	 cv::Mat grayTemp, grayDown;
	 cv::Mat realImage( resizedImageSize, CV_32F );
	 cv::Mat imaginaryImage( resizedImageSize, CV_32F );
	imaginaryImage.setTo( 0 );
	cv::Mat combinedImage( resizedImageSize, CV_32FC2 );
	cv::Mat imageDFT;
	cv::Mat logAmplitude;
	cv::Mat angle( resizedImageSize, CV_32F );
	cv::Mat magnitude( resizedImageSize, CV_32F );
	cv::Mat logAmplitude_blur, imageGR;
	cv::Mat	saliencyMap;
	if( image.channels() == 3 )
	{
		cvtColor( image, imageGR, COLOR_BGR2GRAY );
		imageGR.convertTo( grayDown, CV_32F );
		resize( grayDown, realImage, resizedImageSize, 0, 0, INTER_CUBIC );
	}
	else
	{
		image.convertTo( imageGR, CV_32F );
		resize( imageGR, realImage, resizedImageSize, 0, 0, INTER_CUBIC );
	}
	
	mv.push_back( realImage );
	mv.push_back( imaginaryImage );
	merge( mv, combinedImage );
	dft( combinedImage, imageDFT );
	split( imageDFT, mv );

	//-- Get magnitude and phase of frequency spectrum --//
	cartToPolar( mv.at( 0 ), mv.at( 1 ), magnitude, angle, false );
	log( magnitude, logAmplitude );
	//-- Blur log amplitude with averaging filter --//
	blur( logAmplitude, logAmplitude_blur, Size( 5, 5 ), Point( -1, -1 ), BORDER_DEFAULT );

	exp( logAmplitude - logAmplitude_blur, magnitude );
	//-- Back to cartesian frequency domain --//
	polarToCart( magnitude, angle, mv.at( 0 ), mv.at( 1 ), false );
	merge( mv, imageDFT );
	dft( imageDFT, combinedImage, DFT_INVERSE );
	split( combinedImage, mv );

#if 0
	cartToPolar( mv.at( 0 ), mv.at( 1 ), magnitude, angle, false );
	GaussianBlur( magnitude, magnitude, Size( 5, 5 ),0);
	magnitude = magnitude.mul( magnitude );
#else
	mv.at( 0 ) = mv.at( 0 ).mul( mv.at( 0 ) );
	mv.at( 1 ) = mv.at( 1 ).mul( mv.at( 1 ) );
	magnitude = mv.at( 0 ) + mv.at( 1 );
	GaussianBlur( magnitude, magnitude, Size( 5, 5 ),0);
#endif
	
	normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
	resize( magnitude, saliencyMap, image.size(), 0, 0, INTER_CUBIC );
	saliencyMap.convertTo(saliencyMap, CV_8U, 255);

//	cout<<__FUNCTION__<<":"<<"SR Process, time: " << ((getTickCount() - t_sr)*1000 / getTickFrequency()) << " msec"<<endl;
//	cout<<__FUNCTION__<<":"<<"width:"<<saliencyMap.cols<<"	heigh:"<<saliencyMap.rows<<endl;

	return saliencyMap;
}


