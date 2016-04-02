#include <iostream>
#include <ctime>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include "video_grabber_bb.h"



#define  IMG_HEIGHT  480
#define  IMG_WIDTH   640



class HelperOpencv {
public:
   static void MatToCameraImage(const Mat& mat, CameraImage& img)
   {
      img.alloc(mat.rows, mat.cols);
      for(int i = 0; i < mat.rows; i++) {
         for(int j = 0; j < mat.cols; j++)
            img(i,j) = mat.at<uint8_t>(i,j);
      }
   }
   static void CameraImageToMat(const CameraImage& img, Mat& mat)
   {
      // first we need to decrese refcount in mat and release current data
      // otherwise data can be overwriten by Mat::zeros
      // but we dont really care about that right now... so comment out for speed
      // mat.release();

      // now allocate new memery if needed and copy data
      mat = Mat::zeros(img.rows_, img.cols_, CV_8U);
      for(int i = 0; i < img.rows_; i++) {
         for(int j = 0; j < img.cols_; j++)
            mat.at<uchar>(i,j) = img(i,j);
      }
   }
};


void saveStereoImage(Mat& img_left, Mat& img_right, uint32_t frame_num)
{
   cout << "saving current images\n";
   vector<int> qualityType;
   qualityType.push_back(CV_IMWRITE_PNG_COMPRESSION);
   qualityType.push_back(0); // no compression - fastest
   //imwrite(output_folder + imagelist[i*2+k], cropped, qualityType);
   imwrite("out_img/img_left_" + to_string(frame_num) + ".png", img_left, qualityType);
   imwrite("out_img/img_right_" + to_string(frame_num) + ".png", img_right, qualityType);
}

int main()
{
   VideoGrabberBB grabber(0, 0);

   Mat cvimg_left = Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);
   Mat cvimg_right = Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);

   Mat disp_img = Mat::zeros(IMG_HEIGHT, IMG_WIDTH * 2, CV_8U);
   Size imsz_left = cvimg_left.size();
   Size imsz_right = cvimg_right.size();
   Mat left_part(disp_img, Rect(0, 0, imsz_left.width, imsz_left.height));
   Mat right_part(disp_img, Rect(imsz_left.width, 0, imsz_right.width, imsz_right.height));

   CameraImage img_left(IMG_HEIGHT, IMG_WIDTH);
   CameraImage img_right(IMG_HEIGHT, IMG_WIDTH);

   uint32_t frame_num = 0;
   clock_t curr_time;
   clock_t prev_time;
   prev_time = clock();
   while(true) {
      grabber.getStereoImage(img_left, img_right);

      //HelperOpencv::CameraImageToMat(img_left, cvimg_left);
      //HelperOpencv::CameraImageToMat(img_right, cvimg_right);
      //cvimg_left.copyTo(left_part);
      //cvimg_right.copyTo(right_part);
      //imshow("cam image", disp_img);
      //imshow("cam left", cvimg_left);
      //imshow("cam right", cvimg_right);

      frame_num++;
      cout << "frame: " << frame_num << "\n";;
      // measure fps
      if(frame_num % 100 == 0) {
         curr_time = clock();
         float secs = (float)(curr_time - prev_time) / CLOCKS_PER_SEC;
         cout << "FPS: " << frame_num / secs << "\n";
      }

      //char c = (char) waitKey(1);
      //if(c == 27 || c == 'q' || c == 'Q')
      //   break;
      //else if(c == 13)
      //   saveStereoImage(cvimg_left, cvimg_right, frame_num);
   }

   return 0;
}