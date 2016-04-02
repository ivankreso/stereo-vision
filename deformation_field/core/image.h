#ifndef CORE_IMAGE_H_
#define CORE_IMAGE_H_

#include <cstdint>
#include <cassert>

#include <iostream>
#include <algorithm>
#include <vector>

#include "types.h"

namespace core {

class Image {
public:
   Image(int rows=16, int cols=16, int szPix=1);
   Image(uint8_t* data, int rows, int cols, int szPix=1);
   Image(const Image& other);
   Image(Image&& other);
   ~Image();

   Image& operator=(Image other);
   void resize(int rows, int cols, int szPix=1);
   void dealloc();

   uint8_t& operator()(int row, int col);
   uint8_t operator()(int row, int col) const;

public:
   int szBits() const{
     return rows_*cols_*szPixel_;
   }
   Size size() const {
      return Size(cols_, rows_);
   }
   template <class SrcPixel>
   SrcPixel* pbits(){
     assert(sizeof(SrcPixel)==szPixel_);
     return reinterpret_cast<SrcPixel*>(data_);
   }
   template <class SrcPixel>
   SrcPixel const* pcbits() const{
     assert(sizeof(SrcPixel)==szPixel_);
     return reinterpret_cast<SrcPixel const*>(data_);
   }
   template <class SrcPixel>
   SrcPixel at(int row, int col) const{
     assert(sizeof(SrcPixel)==szPixel_);
     return reinterpret_cast<SrcPixel const*>(data_)[row*cols_ + col];
   }

public:
   int rows_;
   int cols_;
   int szPixel_;
   uint8_t* data_;
   int* refcount_;
   int marginx_; // informative only
   int marginy_; // informative only
};


class ImageSet {
public:
   Image smooth_;
   Image gx_;
   Image gy_;
public:
   int rows() const;
   int cols() const;
   int szPixel() const;
   int szBits() const;
public:
  virtual void compute(const Image& src) =0;
  virtual void config(const std::string& conf) =0;
};

class ImageSetExact: public ImageSet
{
public:
  const double sigmaSmoothing_=0.7;
  const double sigmaGradient_=0.7;
  const double kSigma_=3;
public:
  std::vector<double> kernelSmooth_;
  std::vector<double> kernelGrad_;
public:
  virtual void compute(const Image& src);
  virtual void config(const std::string& conf);
  
  ImageSetExact& operator=(ImageSetExact& other);
};

Image equalize(const Image& src, double mymin=-1, double mymax=-1);
std::pair<double,double> findMinMax(const Image& src);

std::string print(const Image& src);
void savepgm(std::string file, const Image& src);
Image loadpgm(std::string file);

Image maketest(int rows, int cols, int delta);
}

#endif
