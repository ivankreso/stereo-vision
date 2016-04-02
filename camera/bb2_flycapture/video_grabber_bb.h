#ifndef VIDEO_GRABBER_BB_H_
#define VIDEO_GRABBER_BB_H_


#include "camera_image.h"


class ImageFormat{
public:
  ImageFormat(int w, int h, int szPix);
public:
  int width() const {return width_;}
  int height() const {return height_;}
  int szPixel() const {return szPixel_;}
  int szLine() const {return szLine_;}
  int szBits() const {return szLine()*height();}
private:
  int width_;
  int height_;
  int szPixel_;
  int szLine_;
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

  double getStereoPair(CameraImage& imgLeft, CameraImage& imgRight);
  // acquires an image pair after a software trigger
  // places the acquired images into imgLeft and imgRight,
  // returns the local time in ms
  double synchronize();
  // synchronizes the camera with a hardware trigger
// data
private:
  ImageFormat fmt_;
};


#endif
