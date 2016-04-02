// run example:
//./calibrator_stereo_extr -lc bb_left.yml -rc bb_right.yml -w 8 -h 6 -nc -nr -s /home/kreso/projects/master_thesis/datasets/bumblebee/monitor_calib/ ../../../../../datasets/bumblebee/monitor_calib.xml

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

#include "calib_helper.h"

using namespace cv;
using namespace std;

static int print_help()
{
  cout <<
    " Given a list of chessboard images, the number of corners (nx, ny)\n"
    " on the chessboards, and a flag: useCalibrated for \n"
    "   calibrated (0) or\n"
    "   uncalibrated \n"
    "     (1: use cvStereoCalibrate(), 2: compute fundamental\n"
    "         matrix separately) stereo. \n"
    " Calibrate the cameras and display the\n"
    " rectified results along with the computed disparity images.   \n" << endl;
  cout << "Usage:\n ./calibrator -w board_width -h board_height -s src_dir [-nr /*dot not view results*/] <image list XML/YML file>\n" << endl;
  return 0;
}


static void StereoCalib(const string left_cam_file, const string right_cam_file, const string src_dir,
    const vector<string>& imagelist, Size boardSize, bool useCalibrated=true,
    bool showRectified=true, bool showCorners=true)
{
  // read left and right camera matrix data
  FileStorage input_params(left_cam_file, CV_STORAGE_READ);
  Mat cameraMatrix[2], distCoeffs[2];
  input_params["camera_matrix"] >> cameraMatrix[0];
  input_params["distortion_coefficients"] >> distCoeffs[0];
  input_params.release();
  input_params.open(right_cam_file, CV_STORAGE_READ);
  input_params["camera_matrix"] >> cameraMatrix[1];
  input_params["distortion_coefficients"] >> distCoeffs[1];
  input_params.release();
  cout << cameraMatrix[0] << "\n\n";
  cout << cameraMatrix[1] << "\n\n";

  std::string src_dirs[2] = { src_dir + "/left/", src_dir + "/right/" };

  const int maxScale = 2;
  //const double squareSize = .12/5.2;   //  TODO Set this to your actual square size
  //const double squareSize = 0.028;       // 28mm on A4 paper
  const double squareSize = 0.0372;       // 37.2mm on monitor

  // ARRAY AND VECTOR STORAGE:
  vector<vector<Point2f> > imagePoints[2];
  vector<vector<Point3f> > objectPoints;
  Size imageSize;

  int i, j, k, nimages = imagelist.size();

  imagePoints[0].resize(nimages);
  imagePoints[1].resize(nimages);
  vector<string> goodImageList;

  for(i = j = 0; i < nimages; i++)
  {
    for(k = 0; k < 2; k++)
    {
      //const string& filename = src_dir + imagelist[i*2+k];
      const string filename = src_dirs[k] + imagelist[i];
      std::cout << filename << "\n";
      Mat img = imread(filename, 0);
      if(img.empty())
        break;
      if( imageSize == Size() )
        imageSize = img.size();
      else if( img.size() != imageSize )
      {
        cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
        break;
      }
      bool found = false;
      vector<Point2f>& corners = imagePoints[k][j];
      for( int scale = 1; scale <= maxScale; scale++ )
      {
        Mat timg;
        if( scale == 1 )
          timg = img;
        else
          resize(img, timg, Size(), scale, scale);
        found = findChessboardCorners(timg, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
        if(found)
        {
          if( scale > 1 )
          {
            Mat cornersMat(corners);
            cornersMat *= 1./scale;
          }
          break;
        }
      }
      if(showCorners)
      {
        cout << filename << endl;
        Mat cimg, cimg1;
        cvtColor(img, cimg, CV_GRAY2BGR);
        drawChessboardCorners(cimg, boardSize, corners, found);
        double sf = 640./MAX(img.rows, img.cols);
        resize(cimg, cimg1, Size(), sf, sf);
        imshow("corners", cimg1);
        char c = (char)waitKey(500);
        if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
          exit(-1);
        waitKey(0);
      }
      else
        putchar('.');
      if(!found)
        break;
      cornerSubPix(img, corners, Size(11,11), Size(-1,-1), TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 0.01));
    }
    if( k == 2 )
    {
      goodImageList.push_back(imagelist[i]);
      j++;
    }
  }
  cout << j << " pairs have been successfully detected.\n";
  nimages = j;
  if( nimages < 2 )
  {
    cout << "Error: too little pairs to run the calibration\n";
    return;
  }

  imagePoints[0].resize(nimages);
  imagePoints[1].resize(nimages);
  objectPoints.resize(nimages);

  for( i = 0; i < nimages; i++ )
  {
    for( j = 0; j < boardSize.height; j++ )
      for( k = 0; k < boardSize.width; k++ )
        objectPoints[i].push_back(Point3f(j*squareSize, k*squareSize, 0));
  }

  cout << "Running stereo calibration ...\n";


  Mat R, T, E, F;

  double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
      cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1],
      imageSize, R, T, E, F, CV_CALIB_FIX_INTRINSIC, TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));
  cout << "done with RMS error=" << rms << endl;

  // CALIBRATION QUALITY CHECK
  // because the output fundamental matrix implicitly
  // includes all the output information,
  // we can check the quality of calibration using the
  // epipolar geometry constraint: m2^t*F*m1=0
  double err = 0;
  int npoints = 0;
  vector<Vec3f> lines[2];
  for( i = 0; i < nimages; i++ )
  {
    int npt = (int)imagePoints[0][i].size();
    Mat imgpt[2];
    for( k = 0; k < 2; k++ )
    {
      imgpt[k] = Mat(imagePoints[k][i]);
      undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
      computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
    }
    for( j = 0; j < npt; j++ )
    {
      double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
          imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
        fabs(imagePoints[1][i][j].x*lines[0][j][0] +
            imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
      err += errij;
    }
    npoints += npt;
  }
  cout << "average reprojection err = " <<  err/npoints << endl;

  // save intrinsic parameters
  FileStorage fs("calib_params.yml", CV_STORAGE_WRITE);
  if( fs.isOpened() )
  {
    fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] << "M2" << cameraMatrix[1]
    << "D2" << distCoeffs[1] << "R" << R << "T" << T << "E" << E << "F" << F;
    fs.release();
  }
  else
    cout << "Error: can not save the intrinsic parameters\n";

  Mat R1, R2, P1, P2, Q;
  Rect validRoi[2];

  double alpha = 0.0; // 0 to zoom to ROI, 1 not to crop...
  stereoRectify(cameraMatrix[0], distCoeffs[0],
      cameraMatrix[1], distCoeffs[1],
      imageSize, R, T, R1, R2, P1, P2, Q,
      CALIB_ZERO_DISPARITY, alpha, imageSize, &validRoi[0], &validRoi[1]);

  fs.open("calib_rectif_params.yml", CV_STORAGE_WRITE);
  if( fs.isOpened() )
  {
    fs << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
    fs.release();
  }
  else
    cout << "Error: can not save the extrinsic parameters\n";

  // OpenCV can handle left-right
  // or up-down camera arrangements
  bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

  // COMPUTE AND DISPLAY RECTIFICATION - only for visual demonstration
  if( !showRectified )
    return;

  Mat rmap[2][2];
  // IF BY CALIBRATED (BOUGUET'S METHOD)
  //   if( useCalibrated )
  //   {
  //      // we already computed everything
  //   }
  //   // OR ELSE HARTLEY'S METHOD - just for uncalibrated cameras
  //   else
  //      // use intrinsic parameters of each camera, but
  //      // compute the rectification transformation directly
  //      // from the fundamental matrix
  //   {
  //      vector<Point2f> allimgpt[2];
  //      for( k = 0; k < 2; k++ )
  //      {
  //         for( i = 0; i < nimages; i++ )
  //            std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
  //      }
  //      F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
  //      Mat H1, H2;
  //      stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
  //
  //      R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
  //      R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
  //      P1 = cameraMatrix[0];
  //      P2 = cameraMatrix[1];
  //   }

  //Precompute maps for cv::remap()
  initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

  Mat canvas;
  double sf;
  int w, h;
  if( !isVerticalStereo )
  {
    sf = 600./MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width*sf);
    h = cvRound(imageSize.height*sf);
    canvas.create(h, w*2, CV_8UC3);
  }
  else
  {
    sf = 300./MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width*sf);
    h = cvRound(imageSize.height*sf);
    canvas.create(h*2, w, CV_8UC3);
  }

  for(i = 0; i < nimages; i++) {
    for(k = 0; k < 2; k++) {
      std::cout << src_dirs[k] + goodImageList[i] << std::endl;
      Mat img = imread(src_dirs[k] + goodImageList[i], 0), rimg, cimg;
      remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
      cvtColor(rimg, cimg, CV_GRAY2BGR);
      Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
      resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
      if( useCalibrated )
      {
        Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
            cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
        rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
        // TODO
      }
    }

    if( !isVerticalStereo )
      for( j = 0; j < canvas.rows; j += 16 )
        line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
    else
      for( j = 0; j < canvas.cols; j += 16 )
        line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);
    // for demo imwrite("rectif_demo.png", canvas);
    char c = (char)waitKey();
    if( c == 27 || c == 'q' || c == 'Q' )
      break;
  }
}


static bool readStringList( const string& filename, vector<string>& l )
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
  Size boardSize;
  string src_dir;
  string left_cam_file, right_cam_file;
  bool showRectified = true;
  bool showCorners = true;

  for( int i = 1; i < argc; i++ )
  {
    if( string(argv[i]) == "-w" )
    {
      if( sscanf(argv[++i], "%d", &boardSize.width) != 1 || boardSize.width <= 0 )
      {
        cout << "invalid board width" << endl;
        return print_help();
      }
    }
    else if( string(argv[i]) == "-h" )
    {
      if( sscanf(argv[++i], "%d", &boardSize.height) != 1 || boardSize.height <= 0 )
      {
        cout << "invalid board height" << endl;
        return print_help();
      }
    }
    else if( string(argv[i]) == "-lc" )
      left_cam_file = argv[++i];
    else if( string(argv[i]) == "-rc" )
      right_cam_file = argv[++i];
    else if( string(argv[i]) == "-nr" )
      showRectified = false;
    else if(string(argv[i]) == "-nc")
      showCorners = false;
    else if( string(argv[i]) == "--help" )
      return print_help();
    else if(string(argv[i]) == "-s")
      src_dir = argv[++i];
    else if( argv[i][0] == '-' )
    {
      cout << "invalid option " << argv[i] << endl;
      return 0;
    }
  }

  if (boardSize.width <= 0 || boardSize.height <= 0)
  {
    cout << "error: if you specified XML file with chessboards,"
      " you should also specify the board width and height (-w and -h options)" << endl;
    return 0;
  }

  std::vector<std::string> img_files;
  CalibHelper::GetFilesInFolder(src_dir + "/left/", img_files, false);

  StereoCalib(left_cam_file, right_cam_file, src_dir, img_files, boardSize, true, showRectified, showCorners);
  return 0;
}
