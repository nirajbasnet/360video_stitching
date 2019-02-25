#include "Stitch360.h"

Stitch360::Stitch360()
{
    calibration_file= "calib_data.xml";
	scale = 1;
	num_images = 12;
	blend_strength = 5;
	warper_type = "spherical";
	expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
	seam_find_type = "gc_color";
	blend_type = Blender::MULTI_BAND;
	result_name = "stitch_result.jpg";
}
Stitch360::~Stitch360()
{

}
Stitch360::Status Stitch360::calibrateStitch()
{
	return OK;
}

void Stitch360::applyWaveCorrection()
{
	if (do_wave_correct)
	{
		std::vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct_type);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}
}

void Stitch360::setWarperType(std::string warp_type)
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
void Stitch360::setSeamFindType(std::string seam_type)
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
Stitch360::Status Stitch360::doImageRegistration()
{

	std::cout << "Warping images (auxiliary)... " << std::endl;
	int64 t = getTickCount();
	std::vector<UMat> images_warped(num_images);
	std::vector<Size> sizes(num_images);
	std::vector<UMat> masks(num_images);

	// Prepare images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}
	setWarperType(warper_type);
	warper = warper_creator->create(static_cast<float>(warped_image_scale * scale));
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)scale;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;
		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();
		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);

		std::vector<UMat> images_warped_f(num_images);

		for (int i = 0; i < num_images; ++i)
			images_warped[i].convertTo(images_warped_f[i], CV_32F);
		std::cout << "Warping images, time: " << ((getTickCount() - t) /
			getTickFrequency()) << " sec" << std::endl;

		//Compensate exposure errors
		compensator = detail::ExposureCompensator::createDefault(expos_comp_type);
		compensator->feed(corners, images_warped, masks_warped);

		//Find seam masks
		setSeamFindType(seam_find_type);
		seam_finder->find(images_warped_f, corners, masks_warped);

		// Release unused memory
		images_warped.clear();
		images_warped_f.clear();
		masks.clear();

		//Create a blender
		blender = Blender::createDefault(blend_type, false);
		Size dst_sz = resultRoi(corners, sizes).size();
		float blend_width = sqrt(static_cast<float>(dst_sz.area())) *blend_strength / 100.f;
		if (blend_width < 1.f)
			blender = Blender::createDefault(Blender::NO, false);
		else if (blend_type == Blender::MULTI_BAND)
		{
			MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
			mb->setNumBands(static_cast<int>(ceil(log(blend_width) /log(2.)) - 1.));
			std::cout << "Multi-band blender, number of bands: " << mb->numBands() << std::endl;
		}
		else if (blend_type == Blender::FEATHER)
		{
			FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
			fb->setSharpness(1.f / blend_width);
			std::cout << "Feather blender, sharpness: " << fb->sharpness() <<std::endl;
		}
		blender->prepare(corners, sizes);
	}
	std::cout << "Image Registration time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << std::endl;
	return OK;
}
void Stitch360::compose360Stitch()
{
    //read data from calibration file
	UMat img;
	UMat img_warped, img_warped_s;
	UMat dilated_mask, seam_mask, mask, mask_warped;


	std::cout << "Compositing..." << std::endl;
	int64 t = getTickCount();
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		std::cout << "Compositing image #" << indices[img_idx] + 1 << std::endl;

		// Read image and resize it if necessary
		if (abs(scale - 1) > 1e-1)
			resize(input_imgs[img_idx], img, Size(), scale, scale);
		else
			img = input_imgs[img_idx];
		//input_imgs[img_idx].release();    //test it
		Size img_size = img.size();
		Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);

		// Warp the current image
		warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT,img_warped);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST,
			BORDER_CONSTANT, mask_warped);

		// Compensate exposure error step
		compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();
		dilate(masks_warped[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());
		bitwise_and(seam_mask, mask_warped, mask_warped);

		// Blending images step
		blender->feed(img_warped_s, mask_warped, corners[img_idx]);
	}
	Mat result, result_mask;
	blender->blend(result, result_mask);
	std::cout << "Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << std::endl;
	imwrite(result_name, result);

}
