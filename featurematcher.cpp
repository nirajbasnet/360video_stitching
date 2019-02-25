#include "featurematcher.h"


namespace FeatureMatcher{

Matcher::Matcher(const int &NUM_CAMERAS, const double &match_conf_, const double &conf_thresh_, UMat &matcher_mask_){
    this->match_conf_ = match_conf_;
    this->conf_thresh_ = conf_thresh_;
    this->num_images = NUM_CAMERAS;
    this->mask = matcher_mask_.clone();
}

void Matcher::makeMatchPair(vector<MatchesInfo> &pairwise_matches, const int NUM_CAMERAS)
{
    num_images = NUM_CAMERAS;
    near_pairs.clear();
    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.cols == num_images && mask.rows));
    Mat_<uchar> mask_(mask.getMat(ACCESS_READ));
    if (mask_.empty())
        mask_ = Mat::ones(num_images, num_images, CV_8U);

    for (int i = 0; i < num_images; ++i)
        for (int j = i + 1; j < num_images; ++j)
            if (mask_(i, j))
                {
                    near_pairs.push_back(std::make_pair(i, j));
                    pairwise_matches[i*num_images+j].src_img_idx = i;
                    pairwise_matches[i*num_images+j].dst_img_idx = j;
                    pairwise_matches[j*num_images+i].src_img_idx = j;
                    pairwise_matches[j*num_images+i].dst_img_idx = i;
                }
}

void Matcher::match(const std::vector<detail::ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches, int distanceMetric, const bool useMinDistRef, const bool useBFMatcher)
{
    for(int i=0; i<near_pairs.size(); i++)
        for(int j=0; j<pairwise_matches.size();j++)
        {
            if(near_pairs[i] == std::make_pair(pairwise_matches[j].src_img_idx, pairwise_matches[j].dst_img_idx))
                {
                    match(features[near_pairs[i].first], features[near_pairs[i].second], pairwise_matches[near_pairs[i].first*num_images+near_pairs[i].second], distanceMetric, useMinDistRef, useBFMatcher);
                    match(features[near_pairs[i].first], features[near_pairs[i].second], pairwise_matches[near_pairs[i].second*num_images+near_pairs[i].first], distanceMetric, useMinDistRef, useBFMatcher);
                }
        }
}

void Matcher::match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info, int distanceMetric, const bool useMinDistRef, const bool useBFMatcher)
{
    CV_Assert(features1.descriptors.type() == features2.descriptors.type());
    CV_Assert(features2.descriptors.depth() == CV_8U || features2.descriptors.depth() == CV_32F);

    Ptr<cv::DescriptorMatcher> matcher;
    if (useBFMatcher)
    {
        matcher = makePtr<BFMatcher>((int)distanceMetric);
    }
    else
    {
        Ptr<flann::IndexParams> indexParams = makePtr<flann::KDTreeIndexParams>();
        Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>();
        if (features2.descriptors.depth() == CV_8U)
        {
            indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
            searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
        }
        matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
    }

    std::vector< std::vector<DMatch> > pair_matches;
    std::set<std::pair<int,int> > matches;

    // Find 1->2 matches
    matcher->knnMatch(features1.descriptors, features2.descriptors, pair_matches, 2);
    double maxDist = 0;
    double minDist = numeric_limits<float>::max();
    if(useMinDistRef)
    {
        for(int i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            double dist = pair_matches[i][0].distance;
            maxDist = max(maxDist, dist);
            minDist = min(minDist, dist);
            if (pair_matches[i][0].distance < (10*match_conf_) * minDist)
            {
                matches_info.matches.push_back(pair_matches[i][0]);
                matches.insert(std::make_pair(pair_matches[i][0].queryIdx,pair_matches[i][0].trainIdx));
            }
        }
    }
    else
    {
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];
            if (m0.distance < (1.f - match_conf_) * m1.distance)
            {
                matches_info.matches.push_back(m0);
                matches.insert(std::make_pair(m0.queryIdx, m0.trainIdx));
            }
        }
    }

    // Find 2->1 matches
    pair_matches.clear();
    matcher->knnMatch(features2.descriptors, features1.descriptors, pair_matches, 2);

    if(useMinDistRef)
    {
        maxDist = 0;
        minDist = numeric_limits<float>::max();
        for(int i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            double dist = pair_matches[i][0].distance;
            maxDist = max(maxDist, dist);
            minDist = min(minDist, dist);
            if (pair_matches[i][0].distance < (10*match_conf_) * minDist)
            {
                if (matches.find(std::make_pair(pair_matches[i][0].trainIdx, pair_matches[i][0].queryIdx)) == matches.end())
                    matches_info.matches.push_back(DMatch(pair_matches[i][0].trainIdx,pair_matches[i][0].queryIdx,pair_matches[i][0].distance));
            }
        }
    }
    else
    {
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];
            if (m0.distance < (1.f - match_conf_) * m1.distance)
                if (matches.find(std::make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
                    matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
        }
    }

}

void Matcher::appendFeaturesAndMatches(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwiseMatches, vector<ImageFeatures> &features1, vector<MatchesInfo> &pairwiseMatches1)
{
    for(int i=0;i<pairwiseMatches.size();i++)
    {
        for(int j=0;j<pairwiseMatches1[i].matches.size();j++)
        {
            pairwiseMatches1[i].matches[j].queryIdx += features[pairwiseMatches1[i].src_img_idx].keypoints.size();
            pairwiseMatches1[i].matches[j].trainIdx += features[pairwiseMatches1[i].dst_img_idx].keypoints.size();
            pairwiseMatches[i].matches.push_back(pairwiseMatches1[i].matches[j]);
        }
    }
    for(int i=0;i<num_images;i++)
    {
        features[i].keypoints.insert(features[i].keypoints.end(),features1[i].keypoints.begin(), features1[i].keypoints.end());
    }
}

void Matcher::matchParallel(const std::vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches)
{
        for (int i =0; i < near_pairs.size(); ++i)
        {
            int from = near_pairs[i].first;
            int to = near_pairs[i].second;
            int pair_idx = from*num_images + to;

            size_t dual_pair_idx = to*num_images + from;

            pairwise_matches[dual_pair_idx] = pairwise_matches[pair_idx];
            pairwise_matches[dual_pair_idx].src_img_idx = to;
            pairwise_matches[dual_pair_idx].dst_img_idx = from;

            if (!pairwise_matches[pair_idx].H.empty())
                pairwise_matches[dual_pair_idx].H = pairwise_matches[pair_idx].H.inv();

            for (size_t j = 0; j < pairwise_matches[dual_pair_idx].matches.size(); ++j)
                std::swap(pairwise_matches[dual_pair_idx].matches[j].queryIdx,
                          pairwise_matches[dual_pair_idx].matches[j].trainIdx);
        }
}

void Matcher::computeHomography(const vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches, int &num_matches_thresh1_, int &num_matches_thresh2_)
{

    for(int i=0; i<near_pairs.size(); i++)
        for(int j=0; j<pairwise_matches.size();j++)
        {
            if(near_pairs[i] == std::make_pair(pairwise_matches[j].src_img_idx, pairwise_matches[j].dst_img_idx))
                computeHomography(features[near_pairs[i].first], features[near_pairs[i].second], pairwise_matches[j], num_matches_thresh1_, num_matches_thresh2_);
        }

}

void Matcher::computeHomography(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info, int &num_matches_thresh1_, int &num_matches_thresh2_)
{
    // Check if it makes sense to find homography
    if (matches_info.matches.size() < static_cast<size_t>(num_matches_thresh1_))
        return;

    // Construct point-point correspondences for homography estimation
    Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
    for (size_t i = 0; i < matches_info.matches.size(); ++i)
    {
        const DMatch& m = matches_info.matches[i];

        Point2f p = features1.keypoints[m.queryIdx].pt;
        p.x -= features1.img_size.width * 0.5f;
        p.y -= features1.img_size.height * 0.5f;
        src_points.at<Point2f>(0, static_cast<int>(i)) = p;

        p = features2.keypoints[m.trainIdx].pt;
        p.x -= features2.img_size.width * 0.5f;
        p.y -= features2.img_size.height * 0.5f;
        dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
    }

    // Find pair-wise motion
    matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, CV_RANSAC);
    if (matches_info.H.empty() || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
        return;

    // Find number of inliers
    matches_info.num_inliers = 0;
    for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
        if (matches_info.inliers_mask[i])
            matches_info.num_inliers++;

    // These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
    // using Invariant Features"
    matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

    // Set zero confidence to remove matches between too close images, as they don't provide
    // additional information anyway. The threshold was set experimentally.
    matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

    // Check if we should try to refine motion
    if (matches_info.num_inliers < num_matches_thresh2_)
        return;

    // Construct point-point correspondences for inliers only
    src_points.create(1, matches_info.num_inliers, CV_32FC2);
    dst_points.create(1, matches_info.num_inliers, CV_32FC2);
    int inlier_idx = 0;
    for (size_t i = 0; i < matches_info.matches.size(); ++i)
    {
        if (!matches_info.inliers_mask[i])
            continue;

        const DMatch& m = matches_info.matches[i];

        Point2f p = features1.keypoints[m.queryIdx].pt;
        p.x -= features1.img_size.width * 0.5f;
        p.y -= features1.img_size.height * 0.5f;
        src_points.at<Point2f>(0, inlier_idx) = p;

        p = features2.keypoints[m.trainIdx].pt;
        p.x -= features2.img_size.width * 0.5f;
        p.y -= features2.img_size.height * 0.5f;
        dst_points.at<Point2f>(0, inlier_idx) = p;

        inlier_idx++;
    }

    // Rerun motion estimation on inliers only
    matches_info.H = findHomography(src_points, dst_points, RANSAC);
}

Mat Matcher::visualizeKeypointMatches(
    const Mat& imageL,
    const Mat& imageR,
    const MatchesInfo& matches_info,
    const ImageFeatures& features1,
    const ImageFeatures& features2) {

  Mat visualization;
  hconcat(imageL, imageR, visualization);
  vector< pair<Point2f, Point2f> > matchPointPairsLR;
  for(int i=0;i<matches_info.matches.size();i++)
  {
    matchPointPairsLR.push_back(std::make_pair(features1.keypoints[matches_info.matches[i].queryIdx].pt, features2.keypoints[matches_info.matches[i].trainIdx].pt));
  }
  static const Scalar kVisPointColor = Scalar(110, 220, 0);
  for (int i=0; i<matchPointPairsLR.size(); i++) {
    pair<Point2f,Point2f>& pointPair = matchPointPairsLR[i];
    line(
      visualization,
      pointPair.first,
      pointPair.second + Point2f(imageL.cols, 0),
      kVisPointColor,
      1, // thickness
      CV_AA);
  }
  return visualization;
}
} //namespace FeatureMatcher
