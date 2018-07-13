#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <list>

using namespace cv;

namespace OurMogBgs{

	class CV_EXPORTS_W BackgroundSubtractor : public Algorithm
	{
	public:

		virtual ~BackgroundSubtractor();  
		CV_WRAP_AS(apply) virtual void operator()(InputArray image, OutputArray fgmask,
			double learningRate=0);
		virtual void getBackgroundImage(OutputArray backgroundImage) const;
	};
	class CV_EXPORTS_W BackgroundSubtractorMOG3 : public BackgroundSubtractor
	{
	public:
		CV_WRAP BackgroundSubtractorMOG3();
		CV_WRAP BackgroundSubtractorMOG3(int history,  float varThreshold, bool bShadowDetection=true);
		virtual ~BackgroundSubtractorMOG3();
		virtual void operator()(InputArray image, OutputArray fgmask, double learningRate=-1);
		virtual void getBackgroundImage(OutputArray backgroundImage) const;
		virtual void initialize(Size frameSize, int frameType);
	public:
		 /** @brief Returns the number of last frames that affect the background model
    */
    CV_WRAP int getHistory() const ;
    /** @brief Sets the number of last frames that affect the background model
    */
    CV_WRAP void setHistory(int history);

    /** @brief Returns the number of gaussian components in the background model
    */
    CV_WRAP int getNMixtures() const ;
    /** @brief Sets the number of gaussian components in the background model.

    The model needs to be reinitalized to reserve memory.
    */
    CV_WRAP void setNMixtures(int nmixtures) ;//needs reinitialization!

    /** @brief Returns the "background ratio" parameter of the algorithm

    If a foreground pixel keeps semi-constant value for about backgroundRatio\*history frames, it's
    considered background and added to the model as a center of a new component. It corresponds to TB
    parameter in the paper.
     */
    CV_WRAP double getBackgroundRatio() const ;
    /** @brief Sets the "background ratio" parameter of the algorithm
    */
    CV_WRAP void setBackgroundRatio(double ratio) ;

    /** @brief Returns the variance threshold for the pixel-model match

    The main threshold on the squared Mahalanobis distance to decide if the sample is well described by
    the background model or not. Related to Cthr from the paper.
     */
    CV_WRAP double getVarThreshold() const;
    /** @brief Sets the variance threshold for the pixel-model match
    */
    CV_WRAP void setVarThreshold(double varThreshold);

    /** @brief Returns the variance threshold for the pixel-model match used for new mixture component generation

    Threshold for the squared Mahalanobis distance that helps decide when a sample is close to the
    existing components (corresponds to Tg in the paper). If a pixel is not close to any component, it
    is considered foreground or added as a new component. 3 sigma =\> Tg=3\*3=9 is default. A smaller Tg
    value generates more components. A higher Tg value may result in a small number of components but
    they can grow too large.
     */
    CV_WRAP double getVarThresholdGen() const ;
    /** @brief Sets the variance threshold for the pixel-model match used for new mixture component generation
    */
    CV_WRAP void setVarThresholdGen(double varThresholdGen);

    /** @brief Returns the initial variance of each gaussian component
    */
    CV_WRAP double getVarInit() const ;
    /** @brief Sets the initial variance of each gaussian component
    */
    CV_WRAP void setVarInit(double varInit);

    CV_WRAP double getVarMin() const;
    CV_WRAP void setVarMin(double varMin) ;

    CV_WRAP double getVarMax() const ;
    CV_WRAP void setVarMax(double varMax);

    /** @brief Returns the complexity reduction threshold

    This parameter defines the number of samples needed to accept to prove the component exists. CT=0.05
    is a default value for all the samples. By setting CT=0 you get an algorithm very similar to the
    standard Stauffer&Grimson algorithm.
     */
    CV_WRAP double getComplexityReductionThreshold() const;
    /** @brief Sets the complexity reduction threshold
    */
    CV_WRAP void setComplexityReductionThreshold(double ct);

    /** @brief Returns the shadow detection flag

    If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorMOG2 for
    details.
     */
    CV_WRAP bool getDetectShadows() const;
    /** @brief Enables or disables shadow detection
    */
    CV_WRAP void setDetectShadows(bool detectShadows);

    /** @brief Returns the shadow value

    Shadow value is the value used to mark shadows in the foreground mask. Default value is 127. Value 0
    in the mask always means background, 255 means foreground.
     */
    CV_WRAP int getShadowValue() const;
    /** @brief Sets the shadow value
    */
    CV_WRAP void setShadowValue(int value) ;

    /** @brief Returns the shadow threshold

    A shadow is detected if pixel is a darker version of the background. The shadow threshold (Tau in
    the paper) is a threshold defining how much darker the shadow can be. Tau= 0.5 means that if a pixel
    is more than twice darker then it is not shadow. See Prati, Mikic, Trivedi and Cucchiarra,
    *Detecting Moving Shadows...*, IEEE PAMI,2003.
     */
    CV_WRAP double getShadowThreshold() const ;
    /** @brief Sets the shadow threshold
    */
    CV_WRAP void setShadowThreshold(double threshold);

    CV_WRAP void setDetectNFrames(int nframe);

	protected:
		Size frameSize;
		int frameType;
		Mat bgmodel;
		Mat bgmodelUsedModes;
		int nframes;
		int history;
		int nmixtures;
		double varThreshold;
		float backgroundRatio;
		float varThresholdGen; 
		float fVarInit;
		float fVarMin;
		float fVarMax;
		float fCT;
		bool bShadowDetection;
		unsigned char nShadowDetection;
		float fTau;

	};

}
