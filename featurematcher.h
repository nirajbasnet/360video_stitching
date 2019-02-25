#include "opencv2/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include <opencv2/stitching/detail/camera.hpp>
#include "opencv2/stitching/detail/seam_finders.hpp"


#include <opencv2/core/utility.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>


#include "opencv2/stitching/detail/camera.hpp"

#include <algorithm>



using namespace cv;
using namespace cv::detail;
using namespace std;

namespace FeatureMatcher{

class Matcher{
private:
    std::vector<std::pair<int,int> > near_pairs;
    UMat mask;
    int num_images;
    double match_conf_;
    double conf_thresh_;

public:
    Matcher(const int &NUM_CAMERAS, const double &match_conf_, const double &conf_thresh_, UMat &matcher_mask_);

    void makeMatchPair(
        vector<MatchesInfo> &pairwise_matches,
        const int NUM_CAMERAS);

    void match(
        const ImageFeatures &features1,
        const ImageFeatures &features2,
        MatchesInfo &matches_info,
        int distanceMetric,
        const bool useMinDistRef,
        const bool useBFMatcher);

    void match(
        const std::vector<detail::ImageFeatures> &features,
        std::vector<MatchesInfo> &pairwise_matches,
        int distanceMetric,
        const bool useMinDistRef,
        const bool useBFMatcher);

    void matchParallel(
        const std::vector<ImageFeatures> &features,
        vector<MatchesInfo> &pairwise_matches);

    void computeHomography(
        const ImageFeatures &features1,
        const ImageFeatures &features2,
        MatchesInfo &matches_info,
        int &num_matches_thresh1_,
        int &num_matches_thresh2_);

    void computeHomography(
        const vector<ImageFeatures> &features,
        vector<MatchesInfo> &pairwise_matches,
        int &num_matches_thresh1_,
        int &num_matches_thresh2_);

    void appendFeaturesAndMatches(
        vector<ImageFeatures> &features,
        vector<MatchesInfo> &pairwiseMatches,
        vector<ImageFeatures> &features1,
        vector<MatchesInfo> &pairwiseMatches1);

    Mat visualizeKeypointMatches(
        const Mat& imageL,
        const Mat& imageR,
        const MatchesInfo& matches_info,
        const ImageFeatures& features1,
        const ImageFeatures& features2);

    void briskFeatures(Mat image, ImageFeatures &feature, int index, Mat mask);
};

}//namespace FeatureMatcher
