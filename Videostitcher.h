#ifndef VIDEOSTITCHER_H
#define VIDEOSTITCHER_H

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/videoio/videoio.hpp>
#include "featurematcher.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <algorithm>



//-------------- 0  1  2  3  4  5  6  7  8  9 10  11-------------
#define MASK_0  {0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0}  //{1,2,3,5,6}  05
#define MASK_1  {1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}  //{0,2,3,4,7}  13
#define MASK_2  {1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0}  //{0,1,4,5,8}  28
#define MASK_3  {0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0}  //{0,1,6,7,9}
#define MASK_4  {0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0}  //{1,2,7,8,10}
#define MASK_5  {1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1}  //{0,2,6,8,11}
#define MASK_6  {1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1}  //{0,3,5,9,11}
#define MASK_7  {0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0}  //{1,3,4,9,10}
#define MASK_8  {0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1}  //{2,4,5,10,11}
#define MASK_9  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}  //{3,6,7,10,11} 910
#define MASK_10 {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1}  //{4,7,8,9,11}  109
#define MASK_11 {0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0}  //{5,6,8,9,10}   11568

//!                  Fx    Cx      Fy  Cy
#define REFINE_MASK {{1, 0, 1}, {0, 1, 1}, {0, 0, 0}}

using namespace cv;
using namespace cv::detail;
using namespace std;

namespace convertString
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

class CV_EXPORTS AKAZEFeaturesFinder : public detail::FeaturesFinder
{
public:
    AKAZEFeaturesFinder(int descriptor_type = AKAZE::DESCRIPTOR_MLDB,
                        int descriptor_size = 0,
                        int descriptor_channels = 3,
                        float threshold = 0.001f,
                        int nOctaves = 4,
                        int nOctaveLayers = 4,
                        int diffusivity = KAZE::DIFF_PM_G2);

private:
    void find(InputArray image, detail::ImageFeatures &features);

    Ptr<AKAZE> akaze;
};

class VideoStitcher{
   public :

        static void startTimer(void){
            duration_ = static_cast<double>(cv::getTickCount());
        }

        static double getTimeElapsed(void){
            duration_ = static_cast<double>(cv::getTickCount())-duration_;
            duration_ = duration_/(cv::getTickFrequency());
            return duration_;
        }

        enum Status{
            OK = 0,
            ERR_NEED_MORE_IMGS = 1,
            ERR_HOMOGRAPHY_EST_FAIL = 2,
            ERR_CAMERA_PARAMS_ADJUST_FAIL = 3,
            ERROR = 4
        };

        vector<int> stitch_subjects;
        int codec_format;
        std::vector<Mat> input_imgs;                   //input images from video frames of 12 cameras
        static VideoStitcher createDefault();
        void printUsage();
        int parseCmdArgs(int argc, char** argv);

		Status inputVideoFrames();
		Status prepareStitcher(void);
        Status doImageRegistration();
        Status compose360Stitch();

        void setWarperType(std::string warptype);
	    void setSeamFindType(std::string seam_type);
	    void makeMatcherMask();
	    Status writeCalibrationData(string filename, vector<detail::CameraParams> camera_list, vector<Point> corner );
		Status readCalibrationData(string filename, vector<detail::CameraParams> &camera_list, vector<Point> &corner);


   private :
        bool try_use_gpu_;
        bool do_image_resize_;
        bool use_matching_mask_;
        bool do_select_images_;
        bool use_range_matcher_;

        double work_scale_;
        double seam_scale_;
        double compose_scale_;
        double seam_work_aspect_;
        float warped_image_scale;
        double compose_work_aspect_;
        Size work_resolution_;
        Size seam_resolution_;
        Size compose_resolution_;

        float conf_thresh_;
        std::string features_type_;
        std::string bundle_adjustment_cost_func_;
        std::string bundle_adjustment_refine_mask_;
        bool do_wave_correct_;
        detail::WaveCorrectKind wave_correct_kind_;
        bool save_graph_;
        std::string save_graph_to_;
        std::string warp_type_;
        int expos_comp_type_;
        float match_conf_;
        std::string seam_find_type_;
        int blend_type_;
        int timelapse_type_;
        float blend_strength_;
        bool timelapse_;
        int range_width_;
        int NUM_CAMERAS;
        UMat matcher_mask_;
        Mat matcher_mask;
        int compose_width,compose_height;

        VideoWriter stitched_video;
        bool open_video;
        int image_count;

        string result_name_;
        vector<String> img_names_;
		static double duration_;

	    std::vector<Mat> images;                            //copy of input images used for intermediate processing
	    int num_images;                                 //number of input images (optional: can use input_imgs.size() instead)
	    std::string calibration_file;                       //file for writing and reading stitching calibration data
     	std::vector<Size> full_img_sizes;               //full image sizes of all input images
	    vector<Mat> resized_images;
	    std::vector<Mat> seam_est_imgs_;
	    std::vector<int> indices;
	    std::vector<UMat> masks_warped;       //or use masks_warped[NUM_IMAGES] where NUM_IMAGES=12
	    std::vector<Point> corners;           //or use corners[NUM_IMAGES] where NUM_IMAGES=12
	    std::vector<detail::CameraParams> cameras;

	    bool do_image_registration;
	    bool do_calibration;
	    bool read_calibration;
	    Ptr<WarperCreator> warper_creator;
	    Ptr<detail::RotationWarper> warper;
	    Ptr<detail::SeamFinder> seam_finder;
	    Ptr<detail::ExposureCompensator> compensator;
	    Ptr<detail::Blender> blender;
	    vector<Size> sizes;

	    void briskFeatures(Mat image, ImageFeatures &feature, int index, Mat mask);
	    Status findFeatures(vector<detail::ImageFeatures> &features, const string algorithm);
};

#endif // VIDEOSTITCHER_H_INCLUDED
