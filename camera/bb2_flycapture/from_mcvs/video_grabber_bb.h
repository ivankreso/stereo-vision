#ifndef VIDEO_GRABBER_BB_H_
#define VIDEO_GRABBER_BB_H_

#include "FlyCapture2.h"

#include "camera_image.h"

class ImageFormat{
public:
   ImageFormat() {}
   ImageFormat(int w, int h, int szPix) : width_(w), height_(h), szPixel_(szPix) {}
public:
   int width() const {return width_;}
   int height() const {return height_;}
   int szPixel() const {return szPixel_;}
   int szBits() const {return width()*szPixel()*height();}
private:
   int width_;
   int height_;
   int szPixel_;
};


class VideoGrabberBB{
public:
   VideoGrabberBB(int board, int channel);
   // board   ... index of the firewire board
   // channel ... index of the board channel
   // fmt     ... the desired image format
   ~VideoGrabberBB();
public:
   const ImageFormat& fmt();
   // returns the delivered image format

   virtual double getStereoImage(CameraImage& imgLeft, CameraImage& imgRight);
   virtual void splitStereoImage(FlyCapture2::Image& stereo_img, CameraImage& imgLeft, CameraImage& imgRight);


   // acquires an image pair after a software trigger
   // places the acquired images into imgLeft and imgRight,
   // returns the local time in ms
   double synchronize();
   // synchronizes the camera with a hardware trigger
// data
protected:
   void printCameraInfo(FlyCapture2::CameraInfo* pCamInfo);

   ImageFormat fmt_;
   FlyCapture2::Camera cam_;
   FlyCapture2::Error error_;
};


#endif
