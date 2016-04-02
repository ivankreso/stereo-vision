#ifndef CAMERA_IMAGE_H_
#define CAMERA_IMAGE_H_

#include <iostream>
#include <algorithm>
#include <cstdint>


class CameraImage {
public:
   CameraImage(int rows=16, int cols=16);
   CameraImage(uint8_t* data, int rows, int cols);
   CameraImage(const CameraImage& other);
   CameraImage(CameraImage&& other);
   ~CameraImage();

   CameraImage& operator=(CameraImage other);
   uint8_t& operator()(int row, int col);
   uint8_t operator()(int row, int col) const;
   void dealloc();
   void alloc(int rows, int cols);

public:
   int rows_;
   int cols_;
   int szBits_;
   uint8_t* data_;
   int* refcount_;
};


#endif
