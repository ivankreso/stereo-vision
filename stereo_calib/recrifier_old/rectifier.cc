// run with:
// ./rectifier -c calib_params.yml -s ../data/raw_cropped/ -o ../data/rectified_roi/ -nr ../../config_files/img_list.xml

// ./rectifier -c calib_params.yml -s /home/kreso/projects/master_thesis/datasets/bumblebee/raw/ -o /home/kreso/projects/master_thesis/datasets/bumblebee/rectified_roi/ -nr /home/kreso/projects/master_thesis/datasets/bumblebee/img_list.xml

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

static int print_help()
{
   cout << "Usage:\n ./rectifier -i intrinsics.yml -e extrinsics.yml -s src_folder -o output_folder [-nr /*dot not view results*/] <image list XML/YML file>\n" << endl;
   return 0;
}


static void rectify(const vector<string>& imagelist, const string& calib_filename, const string& source_folder, const string& output_folder, bool showRectified=true)
{
   if( imagelist.size() % 2 != 0 )
   {
      cout << "Error: the image list contains odd (non-even) number of elements\n";
      return;
   }

   // dont need this really
   //bool displayCorners = false;		//true;
   //const int maxScale = 2;
   //const float squareSize = 1.f;  // Set this to your actual square size
   // ARRAY AND VECTOR STORAGE:

   int nimages = (int)imagelist.size()/2;



   // reading calibration parameters
   FileStorage fs(calib_filename, CV_STORAGE_READ);
   if(!fs.isOpened())
   {
      cout << "Failed to open file " << calib_filename << endl;
      return;
   }

   // calibration params
   Mat M1, D1, M2, D2, R, T;
   fs["M1"] >> M1;
   fs["D1"] >> D1;
   fs["M2"] >> M2;
   fs["D2"] >> D2;
   fs["R"] >> R;
   fs["T"] >> T;
   fs.release();

   // rectification params
   Mat R1, P1, R2, P2, Q;

   Mat img_tmp = imread(source_folder + imagelist[0], 0);
   Size imageSize = img_tmp.size();
   //img_tmp.deallocate();
   double alpha = 0.0;	   // 0 - will zoom in ROI by default, to disable this set to -1
   Rect validRoi[2];
   stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 
                 alpha, imageSize, &validRoi[0], &validRoi[1]);

   // save camera projection params after rectification in file
   fs.open("rectif_params.yml", CV_STORAGE_WRITE);
   if( fs.isOpened() )
   {
      fs << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
      fs.release();
   }
   else
      cout << "Error: can not save the rectification parameters\n";


   // OpenCV can handle left-right
   // or up-down camera arrangements
   // i will disable this - not needed
   //bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

   // COMPUTE AND DISPLAY RECTIFICATION
   Mat rmap[2][2];
   //Precompute maps for cv::remap()
   initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
   initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

   Mat canvas;
   //double sf;
   int w, h;
   // needed only for vertical stereo and imge scaling for demo
   //    if( !isVerticalStereo )
   //    {
   //        sf = 600./MAX(imageSize.width, imageSize.height);
   //	sf = 1.0;
   //        w = cvRound(imageSize.width*sf);
   //        h = cvRound(imageSize.height*sf);
   //        canvas.create(h, w*2, CV_8UC3);
   //    }
   //    else
   //    {
   //        sf = 300./MAX(imageSize.width, imageSize.height);
   //        w = cvRound(imageSize.width*sf);
   //        h = cvRound(imageSize.height*sf);
   //        canvas.create(h*2, w, CV_8UC3);
   //    }
   // without scaling
   w = cvRound(imageSize.width);
   h = cvRound(imageSize.height);
   canvas.create(h, w*2, CV_8UC3);

   // in the end we crop the ROI area from images
   for(int i = 0; i < nimages; i++)
   {
      if(i % 100 == 0) {
	 printf("%.2f%% completed...\n", ((float)i / nimages) * 100.f);
      }
      for(int k = 0; k < 2; k++)
      {
	 Mat img = imread(source_folder + imagelist[i*2+k], 0), rimg, cimg;
	 remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
	 cvtColor(rimg, cimg, CV_GRAY2BGR);
	 //Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
	 // make demonstration
	 Mat canvasPart = canvas(Rect(w*k, 0, w, h));
	 resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);

	 // display ROI rectangle
	 rectangle(canvasPart, validRoi[k], Scalar(0,0,255), 2, 8);
	 // crop the ROI from image
	 //Mat cropped = rimg(validRoi[k]);
	 // save images
	 std::vector<int> qualityType;
	 qualityType.push_back(CV_IMWRITE_PNG_COMPRESSION);
	 qualityType.push_back(0);
	 //imwrite(output_folder + imagelist[i*2+k], cropped, qualityType);
	 imwrite(output_folder + imagelist[i*2+k], rimg, qualityType);
      }

//      if(!isVerticalStereo)
//	 for(int j = 0; j < canvas.rows; j += 16)
//	    line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
//      else
//	 for(int j = 0; j < canvas.cols; j += 16)
//	    line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

      // display lines
      for(int j = 0; j < canvas.rows; j += 16)
	 line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);

      if(showRectified) {
	 imshow("rectified", canvas);
	 char c = (char)waitKey();
	 if( c == 27 || c == 'q' || c == 'Q' )
	    break;
      }
   }
}


static bool readStringList(const string& filename, vector<string>& l)
{
   l.resize(0);
   FileStorage fs(filename, FileStorage::READ);
   if( !fs.isOpened() )
      return false;
   FileNode n = fs.getFirstTopLevelNode();
   if( n.type() != FileNode::SEQ )
      return false;
   FileNodeIterator it = n.begin(), it_end = n.end();
   for( ; it != it_end; ++it )
      l.push_back((string)*it);
   return true;
}

int main(int argc, char** argv)
{
   string imagelistfn;
   string calib_filename;
   string source_folder;
   string output_folder;
   bool showRectified = true;

   for(int i = 1; i < argc; i++)
   {
      if(strcmp(argv[i], "-c") == 0)
	 calib_filename = argv[++i];
      else if(strcmp(argv[i], "-s") == 0)
	 source_folder = argv[++i];
      else if(strcmp(argv[i], "-o") == 0)
	 output_folder = argv[++i];
      else if(string(argv[i]) == "-nr")
	 showRectified = false;
      else if(string(argv[i]) == "--help")
	 return print_help();
      else if(argv[i][0] == '-')
      {
	 cout << "invalid option " << argv[i] << endl;
	 return print_help();
      }
      else
	 imagelistfn = argv[i];
   }

   if(imagelistfn == "")
   {
      cout << "error: no xml image list given." << endl;
      return print_help();
   }
   if(output_folder == "" || source_folder == "")
   {
      cout << "error: no output or source folder given." << endl;
      return print_help();
   }
   if(calib_filename == "")
   {
      printf("Command-line parameter error: both intrinsic and extrinsic parameters must be specified\n");
      return print_help();
   }

   vector<string> imagelist;
   bool ok = readStringList(imagelistfn, imagelist);
   if(!ok || imagelist.empty())
   {
      cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
      return print_help();
   }

   rectify(imagelist, calib_filename, source_folder, output_folder, showRectified);
   return 0;
}
