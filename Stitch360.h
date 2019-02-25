#ifndef STITCH360_H
#define STITCH360_H


//Wave correct tpye=   WAVE_CORRECT_HORIZ , WAVE_CORRECT_VERT

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"

using namespace cv;
using namespace cv::detail;

class Stitch360
{
private:
	float scale;
	std::vector<cv::UMat> input_imgs;                   //input images from video frames of 12 cameras
	std::vector<Mat> images;                            //copy of input images used for intermediate processing
	unsigned int num_images;                            //number of input images (optional: can use input_imgs.size() instead)
	std::string calibration_file;                       //file for writing and reading stitching calibration data
	std::vector<cv::Size> full_img_sizes;               //full image sizes of all input images
	std::vector<cv::UMat> seam_est_imgs;
	std::vector<int> indices;
	std::vector<UMat> masks_warped;       //or use masks_warped[NUM_IMAGES] where NUM_IMAGES=12
	std::vector<Point> corners;           //or use corners[NUM_IMAGES] where NUM_IMAGES=12
	std::vector<detail::CameraParams> cameras;
	std::vector<Mat> img_homography;                    //Homography matrices for image pairs

	bool do_wave_correct;
	detail::WaveCorrectKind wave_correct_type;

	Ptr<WarperCreator> warper_creator;
	Ptr<detail::RotationWarper> warper;
	Ptr<detail::SeamFinder> seam_finder;
	Ptr<detail::ExposureCompensator> compensator;
	Ptr<detail::Blender> blender;

	std::string warper_type;
	int expos_comp_type;
	std::string seam_find_type;
	int blend_type;
	float blend_strength;
	float warped_image_scale;
	std::string result_name;

public:
	enum Status
	{
		OK = 0,
		ERR_NEED_MORE_IMGS = 1,
		ERR_HOMOGRAPHY_EST_FAIL = 2,
		ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
	};
	Stitch360();
	~Stitch360();
	Status doImageRegistration();
	Status calibrateStitch();
	void compose360Stitch();

	//6.Wave correction module
	void setWaveCorrection(bool flag) { do_wave_correct = flag; }
	void setWaveCorrectType(detail::WaveCorrectKind type) { wave_correct_type = type; }
	void applyWaveCorrection();

	//7.Warp images
	void setWarperType(std::string warptype);
	void setSeamFindType(std::string seam_type);

};

#endif // !STITCH360_H


