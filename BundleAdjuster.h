#include <opencv/cv.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace cv;
using namespace detail;

namespace BA{
/** @brief Base class for all camera parameters refinement methods.
 */
class BundleAdjusterBase : public Estimator
{
public:
    const Mat refinementMask() const { return refinement_mask_.clone(); }
    void setRefinementMask(const Mat &mask)
    {
        CV_Assert(mask.type() == CV_8U && mask.size() == Size(3, 3));
        refinement_mask_ = mask.clone();
    }

    double confThresh() const { return conf_thresh_; }
    void setConfThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }

    TermCriteria termCriteria() { return term_criteria_; }
    void setTermCriteria(const TermCriteria& term_criteria) { term_criteria_ = term_criteria; }

protected:
    /** @brief Construct a bundle adjuster base instance.

    @param num_params_per_cam Number of parameters per camera
    @param num_errs_per_measurement Number of error terms (components) per match
     */
    BundleAdjusterBase(int num_params_per_cam, int num_errs_per_measurement)
        : num_params_per_cam_(num_params_per_cam),
          num_errs_per_measurement_(num_errs_per_measurement)
    {
        setRefinementMask(Mat::ones(3, 3, CV_8U));
        setConfThresh(1.);
        setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 1000, DBL_EPSILON));
    }

    // Runs bundle adjustment
    virtual bool estimate(const std::vector<ImageFeatures> &features,
                          const std::vector<MatchesInfo> &pairwise_matches,
                          std::vector<CameraParams> &cameras);

    /** @brief Sets initial camera parameter to refine.

    @param cameras Camera parameters
     */
    virtual void setUpInitialCameraParams(const std::vector<CameraParams> &cameras) = 0;
    /** @brief Gets the refined camera parameters.

    @param cameras Refined camera parameters
     */
    virtual void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const = 0;
    /** @brief Calculates error vector.

    @param err Error column-vector of length total_num_matches \* num_errs_per_measurement
     */
    virtual void calcError(Mat &err) = 0;
    /** @brief Calculates the cost function jacobian.

    @param jac Jacobian matrix of dimensions
    (total_num_matches \* num_errs_per_measurement) x (num_images \* num_params_per_cam)
     */
    virtual void calcJacobian(Mat &jac) = 0;

    // 3x3 8U mask, where 0 means don't refine respective parameter, != 0 means refine
    Mat refinement_mask_;

    int num_images_;
    int total_num_matches_;

    int num_params_per_cam_;
    int num_errs_per_measurement_;

    const ImageFeatures *features_;
    const MatchesInfo *pairwise_matches_;

    // Threshold to filter out poorly matched image pairs
    double conf_thresh_;

    //Levenberg–Marquardt algorithm termination criteria
    TermCriteria term_criteria_;

    // Camera parameters matrix (CV_64F)
    Mat cam_params_;

    // Connected images pairs
    std::vector<std::pair<int,int> > edges_;
};


/** @brief Implementation of the camera parameters refinement algorithm which minimizes sum of the reprojection
error squares

It can estimate focal length, aspect ratio, principal point.
You can affect only on them via the refinement mask.
 */
class BundleAdjusterReproj : public BundleAdjusterBase
{
public:
    BundleAdjusterReproj() : BundleAdjusterBase(7, 2) {}

private:
    void setUpInitialCameraParams(const std::vector<CameraParams> &cameras);
    void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const;
    void calcError(Mat &err);
    void calcJacobian(Mat &jac);

    Mat err1_, err2_;
};


/** @brief Implementation of the camera parameters refinement algorithm which minimizes sum of the distances
between the rays passing through the camera center and a feature. :

It can estimate focal length. It ignores the refinement mask for now.
 */
class BundleAdjusterRay : public BundleAdjusterBase
{
public:
    BundleAdjusterRay() : BundleAdjusterBase(4, 3) {}

private:
    void setUpInitialCameraParams(const std::vector<CameraParams> &cameras);
    void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const;
    void calcError(Mat &err);
    void calcJacobian(Mat &jac);

    Mat err1_, err2_;
};

} //! namespace BA
