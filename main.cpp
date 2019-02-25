#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include "Videostitcher.h"
#define CAM_NUMBER 3
using namespace cv;
using namespace std;
int keyboard,x;
//Mat frame0,frame1,frame2,frame3,frame4,frame5,frame6,frame7,frame8,frame9,frame10,frame11;
Mat frame[12];


int main()
{
    int frameCounter = 0;
    VideoStitcher stitcher=VideoStitcher::createDefault();
//  namedWindow("Frame3");
//  namedWindow("Frame7");
    string folder = "videos/wetransfer-indoor-corrected";
    VideoCapture capture1(folder + "/0_corrected.MP4");
    VideoCapture capture2(folder + "/1_corrected.MP4");
    VideoCapture capture3(folder + "/2_corrected.MP4");
    VideoCapture capture4(folder + "/3_corrected.MP4");
    VideoCapture capture5(folder + "/4_corrected.MP4");
    VideoCapture capture6(folder + "/5_corrected.MP4");
    VideoCapture capture7(folder + "/6_corrected.MP4");
    VideoCapture capture8(folder + "/7_corrected.MP4");
    VideoCapture capture9(folder + "/8_corrected.MP4");
    VideoCapture capture10(folder + "/9_corrected.MP4");
    VideoCapture capture11(folder + "/10_corrected.MP4");
    VideoCapture capture12(folder + "/11_corrected.MP4");
    if(!capture1.isOpened()){
        cout << "Unable to open video file 0: "  << endl;
        return 1 ;
    }
    if(!capture2.isOpened()){
        cout << "Unable to open video file 1: "  << endl;
        return 1;
    }
    if(!capture3.isOpened()){
        cout << "Unable to open video file 2: "  << endl;
        return 1;
    }
    if(!capture4.isOpened()){
        cout << "Unable to open video file 3: "  << endl;
        return 1;
    }
    if(!capture5.isOpened()){
        cout << "Unable to open video file 4: "  << endl;
        return 1;
    }
    if(!capture6.isOpened()){
        cout << "Unable to open video file 5: "  << endl;
        return 1;
    }
    if(!capture7.isOpened()){
        cout << "Unable to open video file 6: "  << endl;
        return 1;
    }
    if(!capture8.isOpened()){
        cout << "Unable to open video file 7: "  << endl;
        return 1;
    }
    if(!capture9.isOpened()){
        cout << "Unable to open video file 8: "  << endl;
        return 1;
    }
    if(!capture10.isOpened()){
        cout << "Unable to open video file 9: "  << endl;
        return 1;
    }
    if(!capture11.isOpened()){
        cout << "Unable to open video file 10: "  << endl;
        return 1;
    }
    if(!capture12.isOpened()){
        cout << "Unable to open video file 11: "  << endl;
        return 1;
    }
    stitcher.codec_format = static_cast<int>(capture2.get(CAP_PROP_FOURCC));
    int64 stitch_time=getTickCount();
    while( (char)keyboard != 'q' && (char)keyboard != 27 )
    {
        //read the current frame
        if((!capture1.read(frame[0])) || (!capture2.read(frame[1])) || (!capture3.read(frame[2])) || (!capture4.read(frame[3])) || (!capture5.read(frame[4])) || (!capture6.read(frame[5])) || (!capture7.read(frame[6])) || (!capture8.read(frame[7])) || (!capture9.read(frame[8]))  || (!capture10.read(frame[9])) || (!capture11.read(frame[10])) || (!capture12.read(frame[11])) ) {
            cout << "Unable to read next frame." << endl;
            cout <<endl<< "Stitching time: " << ((getTickCount() - stitch_time) / getTickFrequency())<< " sec" << endl;
            break;
        }
        //frame[3] = imread("wetransfer-indoor-corrected/3_undistorted.jpg");
        //frame[7] = imread("wetransfer-indoor-corrected/7_undistorted.jpg");
        int stitch_subjects[] = {0,1,2,3,4,5,6,7,8,9};

        stitcher.input_imgs.clear();
        stitcher.stitch_subjects.clear();
        for(int i=0; i<(sizeof(stitch_subjects)/4); i++)
        {
            stitcher.stitch_subjects.push_back(stitch_subjects[i]);
            stitcher.input_imgs.push_back(frame[stitch_subjects[i]]);
        }

        if(frameCounter++ > 3)
        {
            stitcher.inputVideoFrames();
            stitcher.prepareStitcher();
            stitcher.doImageRegistration();
            stitcher.compose360Stitch();
            keyboard = waitKey( 30 );
        }
    }

    capture1.release();
    capture2.release();
    capture3.release();
    capture4.release();
	capture5.release();
	capture6.release();
	capture7.release();
	capture8.release();
	capture9.release();
	capture10.release();
	capture11.release();
	capture12.release();

	waitKey(0);
	return 0;
}
