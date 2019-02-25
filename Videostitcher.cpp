#include "Videostitcher.h"
#include "BundleAdjuster.h"

double VideoStitcher::duration_=0;
AKAZEFeaturesFinder::AKAZEFeaturesFinder(int descriptor_type,
                                         int descriptor_size,
                                         int descriptor_channels,
                                         float threshold,
                                         int nOctaves,
                                         int nOctaveLayers,
                                         int diffusivity)
{
    akaze = AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                          threshold, nOctaves, nOctaveLayers, diffusivity);
}

void AKAZEFeaturesFinder::find(InputArray image, detail::ImageFeatures &features)
{
    CV_Assert((image.type() == CV_8UC3) || (image.type() == CV_8UC1));
    Mat descriptors;
    UMat uimage = image.getUMat();
    akaze->detectAndCompute(uimage, UMat(), features.keypoints, descriptors);
    features.descriptors = descriptors.getUMat(ACCESS_READ);
}

VideoStitcher::Status VideoStitcher::writeCalibrationData(std::string filename, vector<detail::CameraParams> camera_list, vector<Point> corner )
{
	FileStorage fs(filename, FileStorage::WRITE);
	for (int k = 0; k < camera_list.size(); k++)
	{
		string temp = "Camera"+convertString::to_string(k+1)+"_Rotation";
		fs << temp << camera_list[k].R;
		temp = "Camera"+convertString::to_string(k+1)+"_Focal";
		fs << temp<<camera_list[k].focal;
		temp = "Camera"+convertString::to_string(k+1)+"_Matrix_K";
		fs << temp<<camera_list[k].K();
		temp = "Camera"+convertString::to_string(k+1)+"_ppx";
		fs << temp<<camera_list[k].ppx;
		temp = "Camera"+convertString::to_string(k+1)+"_ppy";
		fs << temp<<camera_list[k].ppy;
		//cout <<endl<< temp << camera_list[k].R;
	}
	for (int k = 0; k < corner.size(); k++)
	{
		std::string temp = "Corner" + convertString::to_string(k + 1);
		fs << temp << corner[k];
		//cout <<endl<< temp << corner[k];
	}

	fs.release();
	return OK;
}

VideoStitcher::Status VideoStitcher::readCalibrationData(std::string filename, vector<detail::CameraParams> &camera_list, vector<Point> &corner)
{
	camera_list.clear();
	corner.clear();
	detail::CameraParams temp_camera;
	Point temp_corner;
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "Error opening fie..." << endl;
		return ERROR;
	}
	for (int k = 0; k < NUM_CAMERAS ; k++)    //replace with total number of possible homography matrices
	{
		string temp_string = "Camera" + convertString::to_string(k + 1)+"_Rotation";
		fs[temp_string] >> temp_camera.R;

		temp_string = "Camera"+convertString::to_string(k+1)+"_Focal";
		fs[temp_string] >> temp_camera.focal;
//
//		temp_string = "Camera"+convertString::to_string(k+1)+"_Matrix_K";
//		fs[temp_string] >> temp_camera.K();

        temp_string = "Camera"+convertString::to_string(k+1)+"_ppx";
		fs[temp_string] >> temp_camera.ppx;

		temp_string = "Camera"+convertString::to_string(k+1)+"_ppy";
		fs[temp_string] >> temp_camera.ppy;
		camera_list.push_back(temp_camera);

        string corner_string = "Corner" + convertString::to_string(k + 1);
		fs[corner_string] >> temp_corner;
		corner.push_back(temp_corner);
	}
	fs.release();
	return OK;
}

void VideoStitcher::setWarperType(std::string warp_type)
{
	if (warp_type == "plane")
		warper_creator = makePtr<cv::PlaneWarper>();
	else if (warp_type == "cylindrical")
		warper_creator = makePtr<cv::CylindricalWarper>();
	else if (warp_type == "spherical")
		warper_creator = makePtr<cv::SphericalWarper>();
	else if (warp_type == "fisheye")
		warper_creator = makePtr<cv::FisheyeWarper>();
	else if (warp_type == "stereographic")
		warper_creator = makePtr<cv::StereographicWarper>();
	else if (warp_type == "compressedPlaneA2B1")
		warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
	else if (warp_type == "compressedPlaneA1.5B1")
		warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
	else if (warp_type == "compressedPlanePortraitA2B1")
		warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
	else if (warp_type == "compressedPlanePortraitA1.5B1")
		warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
	else if (warp_type == "paniniA2B1")
		warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
	else if (warp_type == "paniniA1.5B1")
		warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
	else if (warp_type == "paniniPortraitA2B1")
		warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
	else if (warp_type == "paniniPortraitA1.5B1")
		warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
	else if (warp_type == "mercator")
		warper_creator = makePtr<cv::MercatorWarper>();
	else if (warp_type == "transverseMercator")
		warper_creator = makePtr<cv::TransverseMercatorWarper>();

	if (!warper_creator)
	{
		std::cout << "Can't create the following warper '" << warp_type << "'\n";
	}
}

void VideoStitcher::setSeamFindType(std::string seam_type)
{
	if (seam_type == "no")
		seam_finder = makePtr<NoSeamFinder>();
	else if (seam_type == "voronoi")
		seam_finder = makePtr<VoronoiSeamFinder>();
	else if (seam_type == "gc_color")
		seam_finder = makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
	else if (seam_type == "gc_colorgrad")
		seam_finder = makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	else if (seam_type == "dp_color")
		seam_finder = makePtr<DpSeamFinder>(DpSeamFinder::COLOR);
	else if (seam_type == "dp_colorgrad")
		seam_finder = makePtr<DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
	if (!seam_finder) {
		std::cout << "Can't create the following seam finder" << seam_type << std::endl; }
}

void VideoStitcher::printUsage()
{
    cout <<
        "Rotation model images stitcher.\n\n"
        "stitching_detailed img1 img2 [...imgN] [flags]\n\n"
        "Flags:\n"
		"  --calibrate\n"
		"   Calibrate video stitcher and estimate camera parameters"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_cuda (yes|no)\n"
        "      Try to use CUDA. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --work_pixel (360|480|720) <int>\n"
        "      Resolution height of image. Default is 480"
        "  --features (surf|orb)\n"
        "      Type of features used for images matching. The default is surf.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (reproj|ray)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n"
        "  --timelapse (as_is|crop) \n"
        "      Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names.\n"
        "  --rangewidth <int>\n"
        "      uses range_width to limit number of images to match with.\n";
}

int VideoStitcher::parseCmdArgs(int argc, char** argv)
{
    if(argc == 1)
    {
        printUsage();
        return -1;
    }
    for(int i=1; i<argc; ++i)
    {
        if(string(argv[i]) == "--help"||string(argv[i])=="/?")
        {
            printUsage();
            return -1;
        }
		else if (string(argv[i]) == "--calibrate")
		{
			do_calibration = true;
			i++;
		}
        else if(string(argv[i]) == "--work_megapix")
        {
//            work_megapix_ = atoi(argv[i+1]);
            i++;
        }
        else if(string(argv[i]) == "--work_pixel")
        {
           // work_pixel_ = atoi(argv[i+1]);
            i++;
        }
        else if (string(argv[i]) == "--seam_megapix")
        {
           // seam_megapix_ = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--compose_megapix")
        {
         //   compose_megapix_ = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--result")
        {
            result_name_ = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--features")
        {
            features_type_ = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--match_confidence")
        {
            match_conf_ = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--confidence_threshold")
        {
            conf_thresh_ = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--bundle_adjustment")
        {
            bundle_adjustment_cost_func_ = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--ba_refine_mask")
        {
            bundle_adjustment_refine_mask_ = argv[i + 1];
            if (bundle_adjustment_refine_mask_.size() != 5)
            {
                cout << "Incorrect refinement mask length.\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--wave_correct")
        {
            if (string(argv[i + 1]) == "no")
                do_wave_correct_ = false;
            else if (string(argv[i + 1]) == "horiz")
            {
                do_wave_correct_ = true;
                wave_correct_kind_ = detail::WAVE_CORRECT_HORIZ;
            }else if (string(argv[i + 1]) == "vert")
            {
                do_wave_correct_ = true;
                wave_correct_kind_ = detail::WAVE_CORRECT_VERT;
            }else
            {
                cout << "Bad --wave_correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--save_graph")
        {
            save_graph_ = true;
            save_graph_to_ = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--warp")
        {
            warp_type_ = string(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp")
        {
            if (string(argv[i + 1]) == "no")
                expos_comp_type_ = detail::ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp_type_ = detail::ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp_type_ = detail::ExposureCompensator::GAIN_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--seam")
        {
            if (string(argv[i + 1]) == "no" ||
                string(argv[i + 1]) == "voronoi" ||
                string(argv[i + 1]) == "gc_color" ||
                string(argv[i + 1]) == "gc_colorgrad" ||
                string(argv[i + 1]) == "dp_color" ||
                string(argv[i + 1]) == "dp_colorgrad")
                seam_find_type_ = argv[i + 1];
            else
            {
                cout << "Bad seam finding method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend")
        {
            if (string(argv[i + 1]) == "no")
                blend_type_ = detail::Blender::NO;
            else if (string(argv[i + 1]) == "feather")
                blend_type_ = detail::Blender::FEATHER;
            else if (string(argv[i + 1]) == "multiband")
                blend_type_ = detail::Blender::MULTI_BAND;
            else
            {
    vector<pair<int,int> > near_pairs;                cout << "Bad blending method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--timelapse")
        {
            timelapse_ = true;

            if (string(argv[i + 1]) == "as_is")
                timelapse_type_ = detail::Timelapser::AS_IS;
            else if (string(argv[i + 1]) == "crop")
                timelapse_type_ = detail::Timelapser::CROP;
            else
            {
                cout << "Bad timelapse method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--rangewidth")
        {
            range_width_ = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--blend_strength")
        {
            blend_strength_ = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name_ = argv[i + 1];
            i++;
        }
        else
            img_names_.push_back(argv[i]);
    }
    NUM_CAMERAS = img_names_.size();
    matcher_mask_ = UMat::ones(NUM_CAMERAS, NUM_CAMERAS, CV_8U);
    return 0;
}

VideoStitcher VideoStitcher::createDefault(void)
{
    VideoStitcher videoStitcher;
    videoStitcher.do_image_resize_= true;
    videoStitcher.try_use_gpu_ = false;
    videoStitcher.use_matching_mask_ = true;
    videoStitcher.do_select_images_ = false;
    videoStitcher.use_range_matcher_ = false;

    videoStitcher.work_resolution_= Size(1500,1080);
    videoStitcher.seam_resolution_ = Size(720,640);
    videoStitcher.compose_resolution_= Size(1500,1080);
    videoStitcher.work_scale_=1;
    videoStitcher.seam_scale_=1;
    videoStitcher.compose_scale_=1;
    videoStitcher.warped_image_scale=1;

    videoStitcher.conf_thresh_ = 0.1f;
    videoStitcher.features_type_ = "orb";
    videoStitcher.bundle_adjustment_cost_func_ = "ray";
    videoStitcher.bundle_adjustment_refine_mask_ = "xxxxx";
    videoStitcher.do_wave_correct_ = false;
    videoStitcher.wave_correct_kind_ = detail::WAVE_CORRECT_HORIZ;
    videoStitcher.save_graph_ = true;
    videoStitcher.save_graph_to_="graph.txt";
    videoStitcher.warp_type_ = "spherical";
    videoStitcher.expos_comp_type_ = detail::ExposureCompensator::GAIN;
    videoStitcher.match_conf_ = 0.5f;
    videoStitcher.seam_find_type_ = "gc_color";
    videoStitcher.blend_type_ = detail::Blender::MULTI_BAND;
    videoStitcher.timelapse_type_ = detail::Timelapser::AS_IS;
    videoStitcher.blend_strength_ = 1;
    videoStitcher.timelapse_ = false;
    videoStitcher.range_width_ = -1;
    videoStitcher.NUM_CAMERAS = 12;
    videoStitcher.result_name_ = "resultPanorama.jpg";
	videoStitcher.calibration_file = "calib_data.xml";
	videoStitcher.do_image_registration = true;
	videoStitcher.do_calibration = true;
	videoStitcher.read_calibration = true;
	videoStitcher.open_video=true;
	videoStitcher.image_count=1;

    return videoStitcher;

}

void VideoStitcher::makeMatcherMask()
{
    char mask[12][12] = { MASK_0, MASK_1, MASK_2, MASK_3, MASK_4, MASK_5, MASK_6, MASK_7, MASK_8, MASK_9, MASK_10, MASK_11 };
    //UMat temp((sizeof(stich_subjects)/4), (sizeof(stitch_subjects)/4), CV_8U, cv::Scalar(0));
    uchar temp[input_imgs.size()][input_imgs.size()];
    for (int i=0; i<input_imgs.size(); i++)
    {
        for(int j=0; j<input_imgs.size(); j++)
            temp[i][j]= mask[stitch_subjects[i]][stitch_subjects[j]];
    }

    matcher_mask_ = (Mat(input_imgs.size(), input_imgs.size(), CV_8U, temp).getUMat(ACCESS_RW)).clone();
    cout<<matcher_mask_.getMat(ACCESS_READ);
}

VideoStitcher::Status VideoStitcher::inputVideoFrames()
{
	NUM_CAMERAS = static_cast<int>(input_imgs.size());
	makeMatcherMask();
	cout<<"Number of Cameras="<<NUM_CAMERAS<<endl;
	return OK;
}

VideoStitcher::Status VideoStitcher::findFeatures(vector<detail::ImageFeatures> &features, const string algorithm)
{
    Ptr<detail::FeaturesFinder> finder;
    if (algorithm == "akaze")
    {
        finder = makePtr<AKAZEFeaturesFinder>();
    }
    else if(algorithm == "orb")
    {
        finder = makePtr<detail::OrbFeaturesFinder>();
    }


    Mat raw_image,temp_image;
    vector<Size> raw_image_sizes(NUM_CAMERAS);
    int temp_width,temp_height;
    for(int i=0; i<NUM_CAMERAS; i++)
	{
	        img_names_.push_back(convertString::to_string(i+1));
		    raw_image = input_imgs[i];
            raw_image_sizes[i] = raw_image.size();
            if(raw_image.empty()){
                cout<< "Can't open image" << endl;
                return ERR_HOMOGRAPHY_EST_FAIL;
            }
            work_scale_ = std::min(1.0, std::sqrt((double)work_resolution_.area()/ (double)raw_image.size().area()));
            temp_width=raw_image_sizes[i].width*work_scale_;
            temp_height=raw_image_sizes[i].height*work_scale_;
            resize(raw_image, temp_image, Size(temp_width,temp_height));
            //resize(raw_image, temp_image, work_resolution_);

            resized_images.push_back(temp_image);

            seam_scale_ = std::min(1.0, std::sqrt((double)seam_resolution_.area() / (double)raw_image.size().area()));
            seam_work_aspect_ = seam_scale_/work_scale_;
            temp_width=raw_image_sizes[i].width*seam_scale_;
            temp_height=raw_image_sizes[i].height*seam_scale_;
            resize(raw_image, temp_image, Size(temp_width,temp_height));
            // resize(raw_image, temp_image, seam_resolution_);
            seam_est_imgs_.push_back(temp_image);
            if (algorithm == "akaze")
            {
                (*finder)(resized_images[i], features[i]);
                features[i].img_idx = i;
            }
            else if(algorithm == "orb")
            {
                (*finder)(resized_images[i], features[i]);
                features[i].img_idx = i;
            }
            else if(algorithm == "brisk")
            {
                briskFeatures(resized_images[i], features[i], i, Mat());
            }
            else
            {
                cout<<"Unknown 2D features type :" << features_type_ << endl;
                return ERR_HOMOGRAPHY_EST_FAIL;
            }

    }
    //finder->collectGarbage();
	raw_image.release();
	return OK;
}

VideoStitcher::Status VideoStitcher::prepareStitcher(void)
{
if(do_calibration)
{
    if(NUM_CAMERAS<2)
        return ERR_NEED_MORE_IMGS;

    cout<<"Finding features..";
    vector<detail::ImageFeatures> features(NUM_CAMERAS);
    vector<detail::MatchesInfo> pairwiseMatches(NUM_CAMERAS*NUM_CAMERAS);
    int num_match_thresh1_ = 6;
    int num_match_thresh2_ = 6;

    startTimer();

    //Resize images and Feature Finder
    FeatureMatcher::Matcher matcher(NUM_CAMERAS, match_conf_, conf_thresh_, matcher_mask_);
    matcher.makeMatchPair(pairwiseMatches, NUM_CAMERAS);
    if(findFeatures(features,"brisk")!=OK)
            return ERR_HOMOGRAPHY_EST_FAIL;
    matcher.match(features, pairwiseMatches, NORM_L2, false, false);

    vector<ImageFeatures> features1(NUM_CAMERAS);
    vector<MatchesInfo> pairwiseMatches1(NUM_CAMERAS*NUM_CAMERAS);
    matcher.makeMatchPair(pairwiseMatches1, NUM_CAMERAS);
    Ptr<detail::FeaturesFinder> finder;
    finder = makePtr<OrbFeaturesFinder>();
    for(int i=0;i<NUM_CAMERAS;i++)
    {
        (*finder)(resized_images[i], features1[i]);
        features1[i].img_idx = i;
    }
    matcher.match(features1, pairwiseMatches1, NORM_HAMMING2, false, false);

    vector<ImageFeatures> features2(NUM_CAMERAS);
    vector<MatchesInfo> pairwiseMatches2(NUM_CAMERAS*NUM_CAMERAS);
    matcher.makeMatchPair(pairwiseMatches2, NUM_CAMERAS);
    Ptr<detail::FeaturesFinder> finder2;
    finder2 = makePtr<AKAZEFeaturesFinder>();
    for(int i=0;i<NUM_CAMERAS;i++)
    {
        (*finder2)(resized_images[i], features2[i]);
        features2[i].img_idx = i;
    }
    matcher.match(features2, pairwiseMatches2, NORM_L2, false, false);

    matcher.appendFeaturesAndMatches(features, pairwiseMatches, features1, pairwiseMatches1);
    matcher.appendFeaturesAndMatches(features, pairwiseMatches, features2, pairwiseMatches2);


//    Mat matchView = visualizeKeypointMatches(resized_images[0],resized_images[1],pairwiseMatches[1],features[0],features[1]);
//    namedWindow("matches");
//    imshow("matches",matchView);
//    waitKey(0);


//    for (int i=0;i<pairwiseMatches[5].matches.size();++i)
//    cout <<"match "<< pairwiseMatches[5].matches[i].queryIdx << ": "<< pairwiseMatches[5].matches[i].trainIdx<< endl;

    matcher.computeHomography(features,pairwiseMatches,num_match_thresh1_,num_match_thresh2_);
    matcher.matchParallel(features, pairwiseMatches);

//    cout << "Time to resize and find features :" << getTimeElapsed() << endl << endl;
//
//    // Features Matching
//    cout << "Pairwise matching" << endl;
//    startTimer();

//    cout<<matcher_mask_.getMat(ACCESS_READ);
//    if(!use_range_matcher_)
//    {
//        detail::BestOf2NearestMatcher matcher(false, match_conf_);
//        matcher(features, pairwiseMatches, matcher_mask_);
//        matcher.collectGarbage();
//    }
//    else
//    {
//        detail::BestOf2NearestRangeMatcher matcher(false, match_conf_);
//        matcher(features, pairwiseMatches, matcher_mask_);
//        matcher.collectGarbage();
//    }
//    cout << "Pairwise matching time :" << getTimeElapsed() << endl;

    for(int i=0;i<pairwiseMatches.size();++i)
    {
     cout<< endl <<pairwiseMatches[i].src_img_idx<<"**"<<pairwiseMatches[i].dst_img_idx<<"  Pairwise match confidence="<< pairwiseMatches[i].confidence << endl;
    }

    if (save_graph_)
    {
        cout<<endl<<"Saving matches graph..."<<endl;
        std::ofstream f("graph.txt", std::ofstream::out);
        f << matchesGraphAsString(img_names_, pairwiseMatches, conf_thresh_);
        f.close();
    }


    //Select images and matches subset to build panorama if selected
    indices.clear();
    indices.resize(NUM_CAMERAS);
    if(do_select_images_)
    {
        indices.clear();
        indices = leaveBiggestComponent(features, pairwiseMatches, conf_thresh_);
        vector<Mat> image_subset;
        vector<Size> raw_image_sizes_subset;
        for (uint8_t i=0; i<indices.size();++i)
        {
			image_subset.push_back(resized_images[i]);
        }
		resized_images = image_subset;
    }
    else
    {
        for(uint8_t i=0; i<indices.size(); ++i)
        {
            indices[i] = i;                                //---------------------//
        }
    }

    //Estimate camera parameters rough
    detail::HomographyBasedEstimator estimator;
    if(!estimator(features, pairwiseMatches, cameras))
    {
        cout << "Homography estimation failed." << endl;
        return ERR_HOMOGRAPHY_EST_FAIL;
    }
    for (uint8_t i=0; i<cameras.size(); ++i)
    {
        Mat rotation;
        cameras[i].R.convertTo(rotation, CV_32F);
        cameras[i].R = rotation;
    }

    //Refine camera parameters globallymatches_subset

    uchar temp[3][3] = REFINE_MASK;
    Mat refine_mask = (Mat(3, 3, CV_8U, temp)).clone();

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (bundle_adjustment_cost_func_ == "reproj")
        adjuster = makePtr<detail::BundleAdjusterReproj>();
    else
        adjuster = makePtr<detail::BundleAdjusterRay>();

    adjuster->setConfThresh(conf_thresh_);
    adjuster->setRefinementMask(refine_mask);


    if(!(*adjuster)(features, pairwiseMatches, cameras))
    {
        cout << "Camera parameters adjusting failed." << endl;
        return ERR_CAMERA_PARAMS_ADJUST_FAIL;
    }
}
	return OK;
}

VideoStitcher::Status VideoStitcher::doImageRegistration()
{
	if (do_image_registration||do_calibration)
		{
		//Find median focal length
		vector<double> focals;
		for (uint8_t i = 0; i<cameras.size(); ++i)
		{
			cout << "Camera #" << indices[i] + 1 << ":" << endl << cameras[i].K() << endl;
			focals.push_back(cameras[i].focal);
		}
		sort(focals.begin(), focals.end());

		if (focals.size() % 2 == 1)
			warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
		else
			warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

		cout << endl << "warped_image_scale_ :" << warped_image_scale << endl;

        if (do_wave_correct_)
        {
            std::vector<Mat> rmats;
            for (size_t i = 0; i < cameras.size(); ++i)
                rmats.push_back(cameras[i].R.clone());
            detail::waveCorrect(rmats, wave_correct_kind_);
            for (size_t i = 0; i < cameras.size(); ++i)
                cameras[i].R = rmats[i];
        }
        cout<<endl<<"Seam size="<<seam_est_imgs_[0].size();

		cout <<endl<< "Warping images (auxiliary)... " << endl;
		startTimer();
		//vector<Point> corners(NUM_CAMERAS);
		vector<UMat> images_warped(NUM_CAMERAS);
		sizes.resize(NUM_CAMERAS);
		vector<UMat> masks(NUM_CAMERAS);
		UMat temp_masks_warped;

        // Prepare image masks
		for (uint8_t i = 0; i < NUM_CAMERAS; ++i)
		{
			masks[i].create(seam_est_imgs_[i].size(), CV_8U);
			masks[i].setTo(Scalar::all(255));
		}

		// Map projections
		setWarperType(warp_type_);

		warper = warper_creator->create(float(warped_image_scale*seam_work_aspect_));                 //------------//

		for (uint8_t i = 0; i < NUM_CAMERAS; ++i)
		{
		    Mat_<float> K;
			cameras[i].K().convertTo(K, CV_32F);
			K(0, 0) *= seam_work_aspect_;
			K(0, 2) *= seam_work_aspect_;
			K(1, 1) *= seam_work_aspect_;
			K(1, 2) *= seam_work_aspect_;

			corners.push_back(warper->warp(seam_est_imgs_[i], K, cameras[i].R, INTER_LINEAR, BORDER_CONSTANT, images_warped[i]));
			sizes[i] = images_warped[i].size();
			warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, temp_masks_warped);
			masks_warped.push_back(temp_masks_warped);
		}

		vector<UMat> images_warped_f(NUM_CAMERAS);
		for (uint8_t i = 0; i < NUM_CAMERAS; ++i)
		{
			images_warped[i].convertTo(images_warped_f[i], CV_32F);
		}
        cout << "Warping images, time :" << getTimeElapsed() << endl;
        cout<<endl<<"corners="<<corners[0]<<"**"<<corners[1]<<"**"<<corners[2];
		// Compensate Exposure errors
        cout <<endl<<"Compensating exposure errors... :"<<endl;
		compensator = detail::ExposureCompensator::createDefault(expos_comp_type_);
		compensator->feed(corners, images_warped, masks_warped);

		//Find seam masks
		cout<<"Finding seam masks..."<<endl;
		setSeamFindType(seam_find_type_);
		seam_finder->find(images_warped_f, corners, masks_warped);

		//resized_images.clear();
        seam_est_imgs_.clear();
		images_warped.clear();
		images_warped_f.clear();
		masks.clear();

        // Read image and resize it if necessary
        compose_scale_ = std::min(1.0, std::sqrt((double)compose_resolution_.area()/ (double)input_imgs[0].size().area()));

        // Compute relative scales
        compose_work_aspect_ = compose_scale_ / work_scale_;
        compose_width=input_imgs[0].size().width*compose_scale_;
        compose_height=input_imgs[0].size().height*compose_scale_;

        // Update warped image scale
        warped_image_scale *= static_cast<float>(compose_work_aspect_);
        warper = warper_creator->create(warped_image_scale);

        // Update corners and sizes
        for (int i = 0; i < NUM_CAMERAS; ++i)
        {
            // Update intrinsics
            cameras[i].focal *= compose_work_aspect_;
            cameras[i].ppx *= compose_work_aspect_;
            cameras[i].ppy *= compose_work_aspect_;

            // Update corner and size
            Size sz = input_imgs[i].size();
            if (std::abs(compose_scale_ - 1) > 1e-1)
            {
                sz.width = cvRound(input_imgs[i].size().width * compose_scale_);
                sz.height = cvRound(input_imgs[i].size().height * compose_scale_);
            }

            Mat K;
            cameras[i].K().convertTo(K, CV_32F);
            Rect roi = warper->warpRoi(sz, K, cameras[i].R);
            corners[i] = roi.tl();
            sizes[i] = roi.size();
        }



		//Create a blender
        blender = Blender::createDefault(blend_type_, false);
		Size destinationSize = detail::resultRoi(corners, sizes).size();
		float blendWidth = sqrt(static_cast<float>(destinationSize.area()))*blend_strength_ / 100.f;
		cout<<"blendwidth="<<blendWidth;
		if (blendWidth < 1.f)
			blender = detail::Blender::createDefault(detail::Blender::NO, false);
		else if (blend_type_ == detail::Blender::MULTI_BAND)
		{
			detail::MultiBandBlender* mb = dynamic_cast<detail::MultiBandBlender*>(blender.get());
			mb->setNumBands(static_cast<int>(ceil(log(blendWidth) / log(2.)) - 1.));
			cout <<endl<< "Multi-band blender, number of bands: " << mb->numBands() ;
		}
		else if (blend_type_ == detail::Blender::FEATHER)
		{
			detail::FeatherBlender* fb = dynamic_cast<detail::FeatherBlender*>(blender.get());
			fb->setSharpness(1.f / blendWidth);
			cout << "Feather blender, sharpness: " << fb->sharpness() <<
				endl;
		}
		for(uint8_t i=0;i<NUM_CAMERAS;++i)
            cout<<endl<< "BLENDING Prepare"<< sizes[i]<<endl;

		//write data from calibration file
		writeCalibrationData(calibration_file, cameras, corners);

		do_image_registration = false;
		do_calibration = false;
		}
    cout<<endl<<"Calibration done,data written to file..."<<endl;
    cout<<cameras.size()<<endl;
	return OK;
}

VideoStitcher::Status VideoStitcher::compose360Stitch()
{
    startTimer();
	std::cout << "Compositing..." << std::endl;
	if (read_calibration)
	{
		std::cout << "Reading calibration data from file..." << std::endl;
		readCalibrationData(calibration_file, cameras, corners);          //read data from calibration file
		read_calibration = false;
	}
	else
	{
		//inputVideoFrames();
	}
	Mat full_img,img;
	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;

    blender->prepare(corners, sizes);
	for (int i = 0; i < NUM_CAMERAS;++i)
	{
		cout <<endl<<"Compositing image #" << i+1 << endl;
		int64 pt = getTickCount();
        full_img=input_imgs[i];
		// Read image and resize it if necessary
        if (std::abs(compose_scale_ - 1) > 1e-1)
            resize(full_img, img, Size(compose_width,compose_height));
        else
            img = full_img;

        full_img.release();
        Size img_size = img.size();
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        cout<<endl<<" resize and conversion time: " << ((getTickCount() - pt) / getTickFrequency())*1000 << " ms";

		// Warp the current image
        pt = getTickCount();
		warper->warp(img, K, cameras[i].R, INTER_LINEAR, BORDER_CONSTANT,img_warped);
        cout<<endl<<" warp the current image time: " << ((getTickCount() - pt) / getTickFrequency())*1000 << " ms";
        cout <<endl<<"warped image size" <<img_warped.size();

		// Warp the current image mask
		pt = getTickCount();
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[i].R, INTER_NEAREST,BORDER_CONSTANT, mask_warped);
        cout<<endl<<" warp the current image mask: " << ((getTickCount() - pt) / getTickFrequency())*1000<< " ms";
        cout <<endl<<"warped mask size"<<mask_warped.size();

    	// Compensate exposure error step
    	pt = getTickCount();
        compensator->apply(i, corners[i], img_warped, mask_warped);
        cout<<endl<<" compensate exposure: " << ((getTickCount() - pt) / getTickFrequency())*1000 << " ms"<<endl;

		 pt = getTickCount();
		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();

        // Make sure seam mask has proper size
    	dilate(masks_warped[i], dilated_mask, Mat());
    	cout <<"masks_warped size"<<masks_warped[i].size()<<endl;
		resize(dilated_mask, seam_mask, mask_warped.size());
		cout <<"dilated mask size"<<dilated_mask.size()<<endl;
        bitwise_and(seam_mask, mask_warped, mask_warped);
        cout<<endl<<" other tasks: " << ((getTickCount() - pt) / getTickFrequency())*1000 << " ms";

		// Blending images step
        pt = getTickCount();
		blender->feed(img_warped_s, mask_warped, corners[i]);
		mask_warped.release();
        cout<<endl<<" feed time: " << ((getTickCount() - pt) / getTickFrequency())*1000 << " ms";
	}

    int64 blend_t = getTickCount();
	Mat result, result_mask,final_result;
	blender->blend(result, result_mask);
	cout<<endl<<"blend time: " << ((getTickCount() - blend_t) / getTickFrequency()) << " ms"<<endl;
	std::cout << "Compositing, time: " << getTimeElapsed()*1000<< " ms" << std::endl;
    result_name_="image"+convertString::to_string(image_count)+".jpg";
    image_count++;
    imwrite(result_name_, result);
    result.convertTo(final_result,CV_8U);

    if(open_video)
    {
    stitched_video.open("stitch.avi" ,CV_FOURCC('M', 'J', 'P', 'G'), 25,final_result.size(), true);
    if (!stitched_video.isOpened())
    {
        cout  << "Could not open the output video for write: "  << endl;
        return ERROR;
    }
    open_video=false;
    }
    stitched_video.write(final_result);
	return OK;
}

void VideoStitcher::briskFeatures(Mat image, ImageFeatures &feature, int index, Mat mask)
{
    Ptr<BRISK> detector = BRISK::create();
    Mat descriptors;
    detector->detectAndCompute(image, mask, feature.keypoints, descriptors);
    feature.descriptors = (descriptors.getUMat(ACCESS_READ)).clone();
    feature.img_idx = index;
    feature.img_size = image.size();
}

