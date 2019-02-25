#include "BundleAdjuster.h"
using namespace std;

namespace {

struct IncDistance
{
    IncDistance(std::vector<int> &vdists) : dists(&vdists[0]) {}
    void operator ()(const GraphEdge &edge) { dists[edge.to] = dists[edge.from] + 1; }
    int* dists;
};


struct CalcRotation
{
    CalcRotation(int _num_images, const std::vector<MatchesInfo> &_pairwise_matches, std::vector<CameraParams> &_cameras)
        : num_images(_num_images), pairwise_matches(&_pairwise_matches[0]), cameras(&_cameras[0]) {}

    void operator ()(const GraphEdge &edge)
    {
        int pair_idx = edge.from * num_images + edge.to;

        Mat_<double> K_from = Mat::eye(3, 3, CV_64F);
        K_from(0,0) = cameras[edge.from].focal;
        K_from(1,1) = cameras[edge.from].focal * cameras[edge.from].aspect;
        K_from(0,2) = cameras[edge.from].ppx;
        K_from(1,2) = cameras[edge.from].ppy;

        Mat_<double> K_to = Mat::eye(3, 3, CV_64F);
        K_to(0,0) = cameras[edge.to].focal;
        K_to(1,1) = cameras[edge.to].focal * cameras[edge.to].aspect;
        K_to(0,2) = cameras[edge.to].ppx;
        K_to(1,2) = cameras[edge.to].ppy;

        Mat R = K_from.inv() * pairwise_matches[pair_idx].H.inv() * K_to;
        cameras[edge.to].R = cameras[edge.from].R * R;
    }

    int num_images;
    const MatchesInfo* pairwise_matches;
    CameraParams* cameras;
};


//////////////////////////////////////////////////////////////////////////////

void calcDeriv(const Mat &err1, const Mat &err2, double h, Mat res)
{
    for (int i = 0; i < err1.rows; ++i)
        res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}

} // namespace


namespace BA{

bool BundleAdjusterBase::estimate(const std::vector<ImageFeatures> &features,
                                  const std::vector<MatchesInfo> &pairwise_matches,
                                  std::vector<CameraParams> &cameras)
{
    LOG_CHAT("Bundle adjustment");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    num_images_ = static_cast<int>(features.size());
    features_ = &features[0];
    pairwise_matches_ = &pairwise_matches[0];

    setUpInitialCameraParams(cameras);

    // Leave only consistent image pairs
    edges_.clear();
    for (int i = 0; i < num_images_ - 1; ++i)
    {
        for (int j = i + 1; j < num_images_; ++j)
        {
            const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];
            if (matches_info.confidence > conf_thresh_)
                edges_.push_back(std::make_pair(i, j));
        }
    }

    // Compute number of correspondences
    total_num_matches_ = 0;
    for (size_t i = 0; i < edges_.size(); ++i)
        total_num_matches_ += static_cast<int>(pairwise_matches[edges_[i].first * num_images_ +
                                                                edges_[i].second].num_inliers);

    CvLevMarq solver(num_images_ * num_params_per_cam_,
                     total_num_matches_ * num_errs_per_measurement_,
                     term_criteria_);

    Mat err, jac;
    CvMat matParams = cam_params_;
    cvCopy(&matParams, solver.param);

    int iter = 0;
    for(;;)
    {
        const CvMat* _param = 0;
        CvMat* _jac = 0;
        CvMat* _err = 0;

        bool proceed = solver.update(_param, _jac, _err);

        cvCopy(_param, &matParams);

        if (!proceed || !_err)
            break;

        if (_jac)
        {
            calcJacobian(jac);
            CvMat tmp = jac;
            cvCopy(&tmp, _jac);
        }

        if (_err)
        {
            calcError(err);
            LOG_CHAT(".");
            iter++;
            CvMat tmp = err;
            cvCopy(&tmp, _err);
        }
    }

    LOGLN_CHAT("");
    LOGLN_CHAT("Bundle adjustment, final RMS error: " << std::sqrt(err.dot(err) / total_num_matches_));
    LOGLN_CHAT("Bundle adjustment, iterations done: " << iter);

    // Check if all camera parameters are valid
    bool ok = true;
    for (int i = 0; i < cam_params_.rows; ++i)
    {
        if (cvIsNaN(cam_params_.at<double>(i,0)))
        {
            ok = false;
            break;
        }
    }
    if (!ok)
        return false;

    obtainRefinedCameraParams(cameras);

    // Normalize motion to center image
    Graph span_tree;
    std::vector<int> span_tree_centers;
    findMaxSpanningTree(num_images_, pairwise_matches, span_tree, span_tree_centers);
    std::cout << endl<< "span tree centers " << span_tree_centers[0] << endl;
    Mat R_inv = cameras[span_tree_centers[0]].R.inv();
    for (int i = 0; i < num_images_; ++i)
        cameras[i].R = R_inv * cameras[i].R;

    LOGLN_CHAT("Bundle adjustment, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    return true;
}


//////////////////////////////////////////////////////////////////////////////

void BundleAdjusterReproj::setUpInitialCameraParams(const std::vector<CameraParams> &cameras)
{
    cam_params_.create(num_images_ * 7, 1, CV_64F);
    SVD svd;
    for (int i = 0; i < num_images_; ++i)
    {
        cam_params_.at<double>(i * 7, 0) = cameras[i].focal;
        cam_params_.at<double>(i * 7 + 1, 0) = cameras[i].ppx;
        cam_params_.at<double>(i * 7 + 2, 0) = cameras[i].ppy;
        cam_params_.at<double>(i * 7 + 3, 0) = cameras[i].aspect;

        svd(cameras[i].R, SVD::FULL_UV);
        Mat R = svd.u * svd.vt;
        if (determinant(R) < 0)
            R *= -1;

        Mat rvec;
        Rodrigues(R, rvec);
        CV_Assert(rvec.type() == CV_32F);
        cam_params_.at<double>(i * 7 + 4, 0) = rvec.at<float>(0, 0);
        cam_params_.at<double>(i * 7 + 5, 0) = rvec.at<float>(1, 0);
        cam_params_.at<double>(i * 7 + 6, 0) = rvec.at<float>(2, 0);
    }
}


void BundleAdjusterReproj::obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const
{
    for (int i = 0; i < num_images_; ++i)
    {
        cameras[i].focal = cam_params_.at<double>(i * 7, 0);
        cameras[i].ppx = cam_params_.at<double>(i * 7 + 1, 0);
        cameras[i].ppy = cam_params_.at<double>(i * 7 + 2, 0);
        cameras[i].aspect = cam_params_.at<double>(i * 7 + 3, 0);

        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 7 + 4, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 7 + 5, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 7 + 6, 0);
        Rodrigues(rvec, cameras[i].R);

        Mat tmp;
        cameras[i].R.convertTo(tmp, CV_32F);
        cameras[i].R = tmp;
    }
}


void BundleAdjusterReproj::calcError(Mat &err)
{
    err.create(total_num_matches_ * 2, 1, CV_64F);

    int match_idx = 0;
    for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx)
    {
        int i = edges_[edge_idx].first;
        int j = edges_[edge_idx].second;
        double f1 = cam_params_.at<double>(i * 7, 0);
        double f2 = cam_params_.at<double>(j * 7, 0);
        double ppx1 = cam_params_.at<double>(i * 7 + 1, 0);
        double ppx2 = cam_params_.at<double>(j * 7 + 1, 0);
        double ppy1 = cam_params_.at<double>(i * 7 + 2, 0);
        double ppy2 = cam_params_.at<double>(j * 7 + 2, 0);
        double a1 = cam_params_.at<double>(i * 7 + 3, 0);
        double a2 = cam_params_.at<double>(j * 7 + 3, 0);

        double R1[9];
        Mat R1_(3, 3, CV_64F, R1);
        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 7 + 4, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 7 + 5, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 7 + 6, 0);
        Rodrigues(rvec, R1_);

        double R2[9];
        Mat R2_(3, 3, CV_64F, R2);
        rvec.at<double>(0, 0) = cam_params_.at<double>(j * 7 + 4, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(j * 7 + 5, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(j * 7 + 6, 0);
        Rodrigues(rvec, R2_);

        const ImageFeatures& features1 = features_[i];
        const ImageFeatures& features2 = features_[j];
        const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];

        Mat_<double> K1 = Mat::eye(3, 3, CV_64F);
        K1(0,0) = f1; K1(0,2) = ppx1;
        K1(1,1) = f1*a1; K1(1,2) = ppy1;

        Mat_<double> K2 = Mat::eye(3, 3, CV_64F);
        K2(0,0) = f2; K2(0,2) = ppx2;
        K2(1,1) = f2*a2; K2(1,2) = ppy2;

        Mat_<double> H = K2 * R2_.inv() * R1_ * K1.inv();

        for (size_t k = 0; k < matches_info.matches.size(); ++k)
        {
            if (!matches_info.inliers_mask[k])
                continue;

            const DMatch& m = matches_info.matches[k];
            Point2f p1 = features1.keypoints[m.queryIdx].pt;
            Point2f p2 = features2.keypoints[m.trainIdx].pt;
            double x = H(0,0)*p1.x + H(0,1)*p1.y + H(0,2);
            double y = H(1,0)*p1.x + H(1,1)*p1.y + H(1,2);
            double z = H(2,0)*p1.x + H(2,1)*p1.y + H(2,2);

            err.at<double>(2 * match_idx, 0) = p2.x - x/z;
            err.at<double>(2 * match_idx + 1, 0) = p2.y - y/z;
            match_idx++;
        }
    }
}


void BundleAdjusterReproj::calcJacobian(Mat &jac)
{
    jac.create(total_num_matches_ * 2, num_images_ * 7, CV_64F);
    jac.setTo(0);

    double val;
    const double step = 1e-4;

    for (int i = 0; i < num_images_; ++i)
    {
        if (refinement_mask_.at<uchar>(0, 0))
        {
            val = cam_params_.at<double>(i * 7, 0);
            cam_params_.at<double>(i * 7, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7));
            cam_params_.at<double>(i * 7, 0) = val;
        }
        if (refinement_mask_.at<uchar>(0, 2))
        {
            val = cam_params_.at<double>(i * 7 + 1, 0);
            cam_params_.at<double>(i * 7 + 1, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7 + 1, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + 1));
            cam_params_.at<double>(i * 7 + 1, 0) = val;
        }
        if (refinement_mask_.at<uchar>(1, 2))
        {
            val = cam_params_.at<double>(i * 7 + 2, 0);
            cam_params_.at<double>(i * 7 + 2, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7 + 2, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + 2));
            cam_params_.at<double>(i * 7 + 2, 0) = val;
        }
        if (refinement_mask_.at<uchar>(1, 1))
        {
            val = cam_params_.at<double>(i * 7 + 3, 0);
            cam_params_.at<double>(i * 7 + 3, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7 + 3, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + 3));
            cam_params_.at<double>(i * 7 + 3, 0) = val;
        }
        for (int j = 4; j < 7; ++j)
        {
            val = cam_params_.at<double>(i * 7 + j, 0);
            cam_params_.at<double>(i * 7 + j, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7 + j, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + j));
            cam_params_.at<double>(i * 7 + j, 0) = val;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////

void BundleAdjusterRay::setUpInitialCameraParams(const std::vector<CameraParams> &cameras)
{
    cam_params_.create(num_images_ * 4, 1, CV_64F);
    SVD svd;
    for (int i = 0; i < num_images_; ++i)
    {
        cam_params_.at<double>(i * 4, 0) = cameras[i].focal;

        svd(cameras[i].R, SVD::FULL_UV);
        Mat R = svd.u * svd.vt;
        if (determinant(R) < 0)
            R *= -1;

        Mat rvec;
        Rodrigues(R, rvec);
        CV_Assert(rvec.type() == CV_32F);
        cam_params_.at<double>(i * 4 + 1, 0) = rvec.at<float>(0, 0);
        cam_params_.at<double>(i * 4 + 2, 0) = rvec.at<float>(1, 0);
        cam_params_.at<double>(i * 4 + 3, 0) = rvec.at<float>(2, 0);
    }
}


void BundleAdjusterRay::obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const
{
    for (int i = 0; i < num_images_; ++i)
    {
        cameras[i].focal = cam_params_.at<double>(i * 4, 0);

        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
        Rodrigues(rvec, cameras[i].R);

        Mat tmp;
        cameras[i].R.convertTo(tmp, CV_32F);
        cameras[i].R = tmp;
    }
}


void BundleAdjusterRay::calcError(Mat &err)
{
    err.create(total_num_matches_ * 3, 1, CV_64F);

    int match_idx = 0;
    for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx)
    {
        int i = edges_[edge_idx].first;
        int j = edges_[edge_idx].second;
        double f1 = cam_params_.at<double>(i * 4, 0);
        double f2 = cam_params_.at<double>(j * 4, 0);

        double R1[9];
        Mat R1_(3, 3, CV_64F, R1);
        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
        Rodrigues(rvec, R1_);

        double R2[9];
        Mat R2_(3, 3, CV_64F, R2);
        rvec.at<double>(0, 0) = cam_params_.at<double>(j * 4 + 1, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(j * 4 + 2, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(j * 4 + 3, 0);
        Rodrigues(rvec, R2_);

        const ImageFeatures& features1 = features_[i];
        const ImageFeatures& features2 = features_[j];
        const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];

        Mat_<double> K1 = Mat::eye(3, 3, CV_64F);
        K1(0,0) = f1; K1(0,2) = features1.img_size.width * 0.5;
        K1(1,1) = f1; K1(1,2) = features1.img_size.height * 0.5;

        Mat_<double> K2 = Mat::eye(3, 3, CV_64F);
        K2(0,0) = f2; K2(0,2) = features2.img_size.width * 0.5;
        K2(1,1) = f2; K2(1,2) = features2.img_size.height * 0.5;

        Mat_<double> H1 = R1_ * K1.inv();
        Mat_<double> H2 = R2_ * K2.inv();

        for (size_t k = 0; k < matches_info.matches.size(); ++k)
        {
            if (!matches_info.inliers_mask[k])
                continue;

            const DMatch& m = matches_info.matches[k];

            Point2f p1 = features1.keypoints[m.queryIdx].pt;
            double x1 = H1(0,0)*p1.x + H1(0,1)*p1.y + H1(0,2);
            double y1 = H1(1,0)*p1.x + H1(1,1)*p1.y + H1(1,2);
            double z1 = H1(2,0)*p1.x + H1(2,1)*p1.y + H1(2,2);
            double len = std::sqrt(x1*x1 + y1*y1 + z1*z1);
            x1 /= len; y1 /= len; z1 /= len;

            Point2f p2 = features2.keypoints[m.trainIdx].pt;
            double x2 = H2(0,0)*p2.x + H2(0,1)*p2.y + H2(0,2);
            double y2 = H2(1,0)*p2.x + H2(1,1)*p2.y + H2(1,2);
            double z2 = H2(2,0)*p2.x + H2(2,1)*p2.y + H2(2,2);
            len = std::sqrt(x2*x2 + y2*y2 + z2*z2);
            x2 /= len; y2 /= len; z2 /= len;

            double mult = std::sqrt(f1 * f2);
            err.at<double>(3 * match_idx, 0) = mult * (x1 - x2);
            err.at<double>(3 * match_idx + 1, 0) = mult * (y1 - y2);
            err.at<double>(3 * match_idx + 2, 0) = mult * (z1 - z2);

            match_idx++;
        }
    }
}


void BundleAdjusterRay::calcJacobian(Mat &jac)
{
    jac.create(total_num_matches_ * 3, num_images_ * 4, CV_64F);

    double val;
    const double step = 1e-3;

    for (int i = 0; i < num_images_; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            val = cam_params_.at<double>(i * 4 + j, 0);
            cam_params_.at<double>(i * 4 + j, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 4 + j, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 4 + j));
            cam_params_.at<double>(i * 4 + j, 0) = val;
        }
    }
}

void adjustIndependently(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwiseMatches, vector<CameraParams> &cameras, Ptr<BundleAdjusterBase> adjuster, const UMat &matcher_mask_, vector<int> &stitch_subjects)
{
        vector< vector<CameraParams> > cameras_subset(3);
    vector< vector<MatchesInfo> > matches_subset(3);
    vector< vector<ImageFeatures> > features_subset(3);
    vector< vector<int> > subset_idx(3);

    //! group the cameras on the middle into subset 3 for adjustment
    //! group the bottom and top cameras on subset 0 and 2 respectively and also group their most overlapping image parters
    const int num_images = features.size();
    Mat_<uchar> mask_(matcher_mask_.getMat(ACCESS_READ));
    if (mask_.empty())
        mask_ = Mat::ones(num_images, num_images, CV_8U);

    for(int i=0;i<cameras.size();i++)
    {
        if(stitch_subjects[i]>=3 && stitch_subjects[i]<=8)
        {
            subset_idx[1].push_back(i);
            cameras_subset[1].push_back(cameras[i]);
            features_subset[1].push_back(features[i]);
        }
        else if(stitch_subjects[i]<3)
        {
            subset_idx[0].push_back(i);
            cameras_subset[0].push_back(cameras[i]);
            features_subset[0].push_back(features[i]);
            for(int j=0;(j<stitch_subjects.size());j++)
            {
                if(mask_(i,j) && (stitch_subjects[j]>=3 && stitch_subjects[j]<=8))
                {
                    subset_idx[0].push_back(j);
                    cameras_subset[0].push_back(cameras[j]);
                    features_subset[0].push_back(features[j]);
                }
            }
        }
        else if(stitch_subjects[i]>8)
        {
            subset_idx[2].push_back(i);
            cameras_subset[3].push_back(cameras[i]);
            features_subset[2].push_back(features[i]);
            for(int j=0;j<stitch_subjects.size();j++)
            {
                if(mask_(i,j) && (stitch_subjects[j]>=3 && stitch_subjects[j]<=8))
                {
                    subset_idx[2].push_back(j);
                    cameras_subset[2].push_back(cameras[j]);
                    features_subset[2].push_back(features[j]);
                }
            }
        }
    }

//    matches_subset[0].resize(features_subset[0].size()*features_subset[0].size());
//    matches_subset[1].resize(features_subset[1].size()*features_subset[1].size());
//    matches_subset[2].resize(features_subset[2].size()*features_subset[2].size());


    for(int i=0;i<3;i++)
    {
        matches_subset[i].resize(features_subset[i].size()*features_subset[i].size());
        int num_imgs_subset = features_subset[i].size();
        for(int j=0;j<num_imgs_subset;j++)
        {
            for(int k=0;k<num_imgs_subset;k++)
            {

                if(mask_(subset_idx[i][j],subset_idx[i][k]))
                {
                    matches_subset[i][j*num_imgs_subset+k]=pairwiseMatches[subset_idx[i][j]*num_images+subset_idx[i][k]];
                    matches_subset[i][j*num_imgs_subset+k].src_img_idx = j;
                    matches_subset[i][j*num_imgs_subset+k].dst_img_idx = k;
                }
            }
        }
    }


    for(int i=0;i<features_subset[0].size();i++)
        cout<<"subset pairs 0 "<<subset_idx[0][i]<<endl;
    cout<<endl<<"matches subset check.."<<endl;
    for(int i=0;i<matches_subset[0].size();++i)
    {
     cout<< endl <<matches_subset[0][i].src_img_idx<<"**"<<matches_subset[0][i].dst_img_idx<<"  Pairwise match confidence="<< matches_subset[0][i].confidence << endl;
    }

    (*adjuster)(features_subset[0], matches_subset[0], cameras_subset[0]);
    (*adjuster)(features_subset[1], matches_subset[1], cameras_subset[1]);

    for(int i=0;i<cameras_subset[1].size();i++)
    {
        cameras[subset_idx[1][i]].R = cameras_subset[1][i].R;
    }
    int fifth_img_idx = 0;
    for(int i=0;i<stitch_subjects.size();i++)
    {
        if(stitch_subjects[i]==3)
           fifth_img_idx = i;
    }

    Mat R_ = cameras[fifth_img_idx].R;
    Mat R_inv;

    cout<<"features subset size "<<features_subset[0].size()<<endl;

    for(int i=0;i<features_subset[0].size();i++)
    {
        cout<<"stitch subject "<<stitch_subjects[subset_idx[0][i]]<<endl;
        if(stitch_subjects[subset_idx[0][i]]==3)
        {
            R_inv = cameras_subset[0][i].R.inv();
            break;
        }
    }

    for(int i=0;i<cameras_subset[0].size();i++)
    {
        if(stitch_subjects[subset_idx[0][i]]<3)
            cameras[subset_idx[0][i]].R = R_*(R_inv*cameras_subset[0][i].R);
    }

}
} //! namespace BA
