#include "camera_image.h"

#include <cstring>
#include <utility>

CameraImage::CameraImage(int rows, int cols):
   rows_(rows),
   cols_(cols),
   szBits_(rows*cols),
   data_(new uint8_t[szBits_]),
   refcount_(new int)
{
   *refcount_ = 1;
}

CameraImage::CameraImage(uint8_t* data, int rows, int cols) :
   rows_(rows),
   cols_(cols),
   szBits_(rows*cols),
   data_(data),
   refcount_(nullptr)
{
}

CameraImage::CameraImage(const CameraImage& other):
   rows_(other.rows_),
   cols_(other.cols_),
   szBits_(other.szBits_),
   data_(new uint8_t[other.szBits_]),
   refcount_(new int)
{
   //std::cout << "using copy construcor\n";
   std::memcpy(data_, other.data_, szBits_);
   *refcount_ = 1;
}

CameraImage::CameraImage(CameraImage&& other):
   rows_(other.rows_),
   cols_(other.cols_),
   szBits_(other.szBits_),
   data_(other.data_)
{
   // we have a new reference
   (*refcount_)++;
}

CameraImage::~CameraImage()
{
   // if data is not owned by object dont clean
   if(refcount_ == nullptr)
      return;
   // one reference gone...
   (*refcount_)--;
   // if it was the last reference - clean
   if(*refcount_ == 0)
      dealloc();
}                                                                                 

CameraImage& CameraImage::operator=(CameraImage other)
{
   rows_=other.rows_;
   cols_=other.cols_;
   szBits_=other.szBits_;
   // swap reference counts and data so that "other" can decrement refcount when it leaves block scope
   std::swap(refcount_, other.refcount_);
   std::swap(data_, other.data_);
   return *this;
}

uint8_t& CameraImage::operator()(int row, int col)
{
   return data_[row*cols_ + col];
}

uint8_t CameraImage::operator()(int row, int col) const
{
   return data_[row*cols_ + col];
}

void CameraImage::dealloc()
{
   if(data_ != nullptr) {
      delete[] data_;
      delete refcount_;
      rows_ = 0;
      cols_ = 0;
      szBits_ = 0;
   }
}

void CameraImage::alloc(int rows, int cols)
{
   // decrese ref counter
   (*refcount_)--;
   if(*refcount_ == 0)
      dealloc();
   // alloc new data
   rows_ = rows;
   cols_ = cols;
   szBits_ = rows_ * cols_;
   data_ = new uint8_t[szBits_];
   refcount_ = new int;
   *refcount_ = 1;
}

namespace{
   int test(){
      CameraImage img;
      CameraImage img2(img);
      CameraImage img3;
      img3=img;     
      return 0;
   }
   //static int bla=test();
}

