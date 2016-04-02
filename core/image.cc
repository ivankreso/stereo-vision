#include "image.h"

#include <cstring>
#include <cassert>
#include <cmath>

#include <utility>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>      // std::setprecision

namespace core {

//////////////////////////////////////////////////////////////////
// Image

Image::Image(int rows, int cols, int szPix):
   rows_(rows),
   cols_(cols),
   szPixel_(szPix),
   data_(new uint8_t[szBits()]),
   refcount_(new int),
   marginx_(0), marginy_(0)
{
   *refcount_ = 1;
}

// use as a copy constructor for now because of easier interaction between cv::Mat and Libviso::process
Image::Image(uint8_t* data, int rows, int cols, int szPix) :
   rows_(rows),
   cols_(cols),
   szPixel_(szPix),
   //data_(data), // dont use this please...
   //refcount_(nullptr),
   data_(new uint8_t[szBits()]),
   refcount_(new int),
   marginx_(0), marginy_(0)
{
   std::memcpy(data_, data, szBits());
   *refcount_ = 1;
}

Image::Image(const Image& other):
   rows_(other.rows_),
   cols_(other.cols_),
   szPixel_(other.szPixel_),
   data_(new uint8_t[other.szBits()]),
   refcount_(new int),
   marginx_(other.marginx_), marginy_(other.marginy_)
{
   //std::cout << "using copy constructor\n";
   std::memcpy(data_, other.data_, szBits());
   *refcount_ = 1;
}

Image::Image(Image&& other):
   rows_(other.rows_),
   cols_(other.cols_),
   szPixel_(other.szPixel_),
   data_(other.data_),
   refcount_(other.refcount_),
   marginx_(other.marginx_), marginy_(other.marginy_)
{
   // we have a new reference
   (*refcount_)++;
}

Image::~Image()
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

Image& Image::operator=(Image other)
{
   rows_=other.rows_;
   cols_=other.cols_;
   szPixel_=other.szPixel_;
   marginx_=other.marginx_;
   marginy_=other.marginy_;
   // swap reference counts and data so that "other" can decrement refcount when it leaves block scope
   std::swap(refcount_, other.refcount_);
   std::swap(data_, other.data_);
   return *this;
}


uint8_t& Image::operator()(int row, int col)
{
   return data_[row*cols_ + col];
}

uint8_t Image::operator()(int row, int col) const
{
   return data_[row*cols_ + col];
}


void Image::dealloc()
{
   if(data_ != nullptr) {
      delete[] data_;
      delete refcount_;
      rows_ = 0;
      cols_ = 0;
      szPixel_ = 0;
   }
}

void Image::resize(int rows, int cols, int szPix)
{
   // decrease ref counter
   (*refcount_)--;
   if(*refcount_ == 0)
      dealloc();
   // alloc new data
   rows_ = rows;
   cols_ = cols;
   szPixel_ = szPix;
   data_ = new uint8_t[szBits()];
   refcount_ = new int;
   *refcount_ = 1;
}


////////////////////////////////////////
// createPixel

// integral pixels
template <class Pixel,
  typename std::enable_if<std::is_integral<Pixel>::value>::type* x=nullptr>
Pixel createPixel(double val){
  return int(std::round(val));
}
// float pixels
template <class Pixel,
  typename std::enable_if<std::is_floating_point<Pixel>::value>::type* x=nullptr>
Pixel createPixel(double val){
  return Pixel(val);
}
// other pixels
template <class Pixel,
  typename std::enable_if<!std::is_arithmetic<Pixel>::value>::type* x=nullptr>
Pixel createPixel(double val){
  return Pixel(val);
}

////////////////////////////////////////
// roundPixel

// integral pixels
template <class SrcPixel, 
  typename std::enable_if<std::is_integral<SrcPixel>::value>::type* x=nullptr>
int roundPixel(SrcPixel pix){
  return pix;
}
// float pixels
template <class SrcPixel,
  typename std::enable_if<std::is_floating_point<SrcPixel>::value>::type* x=nullptr>
int roundPixel(SrcPixel pix){
  return int(std::round(pix));
}
// other pixels
template <class SrcPixel,
  typename std::enable_if<!std::is_arithmetic<SrcPixel>::value>::type* x=nullptr>
int roundPixel(SrcPixel pix){
  return pix.round();
}


////////////////////////////////////////
// findMinMax

template <class SrcPixel>
std::pair<SrcPixel,SrcPixel> 
findMinMaxNative(const core::Image& src)
{
  SrcPixel mymax= std::numeric_limits<SrcPixel>::min();
  SrcPixel mymin= std::numeric_limits<SrcPixel>::max();
  for (int j=src.marginy_; j<src.rows_-src.marginy_; ++j){
    SrcPixel const* psrc = src.pcbits<SrcPixel>() + j*src.cols_ + src.marginx_;
    for (int i = src.marginx_; i<src.cols_-src.marginx_; ++i){
      //std::cerr <<(void*)psrc <<"\n";
      mymax=std::max(mymax,*psrc);
      mymin=std::min(mymin,*psrc);
      ++psrc;
    }
  }
  return std::make_pair(mymin, mymax);
}
template <class SrcPixel>
std::pair<double,double> 
findMinMaxWorker(
  const core::Image& src)
{
  auto limits=findMinMaxNative<SrcPixel>(src);
  return std::pair<double,double>(limits.first, limits.second);
}
std::pair<double,double> findMinMax(const Image& src){
  static const std::map<int, decltype(&findMinMaxWorker<int>)>
      M={{1, &findMinMaxWorker<uint8_t>},{4, &findMinMaxWorker<float>}};
  return (*M.at(src.szPixel_))(src);
}


////////////////////////////////////////
// fillBorders

// TODO: check for bugs...
template <class Pixel>
void fillBorders(
  core::Image& img, Pixel val)
{
  Pixel* p = img.pbits<Pixel>();
  for (int j=0; j<img.marginy_; ++j){
    for (int i=0; i<img.cols_; ++i){
      *p++=val;
    }
  }

  for (int j=img.marginy_; j<img.rows_-img.marginy_; ++j){
    Pixel* p = img.pbits<Pixel>() + j*img.cols_;
    for (int i=0; i<img.marginx_; ++i){
      *p++=val;
    }
    p += (img.cols_ -2*img.marginx_);
    for (int i=0; i<img.marginx_; ++i){
      *p++=val;
    }
  }

  p = img.pbits<Pixel>() + (img.rows_-img.marginy_)*img.cols_;
  for (int j=0; j<img.marginy_; ++j){
    for (int i=0; i<img.cols_; ++i){
      *p++=val;
    }
  }
}

////////////////////////////////////////
// equalize

uint8_t clamp(double val){
  return uint8_t(std::round(std::min(255.0, std::max(0.0, val))));
}

template <class SrcPixel>
Image equalizeWorker(
  const core::Image& src,
  double mingiven, double maxgiven)
{  
  auto limits=findMinMaxNative<SrcPixel>(src);
  SrcPixel mymin= (mingiven==-1)? limits.first : mingiven;
  SrcPixel mymax= (maxgiven==-1)? limits.second : maxgiven;
  double factor=255.0/(mymax-mymin);

  Image dst(src.rows_, src.cols_, 1);
  dst.marginx_=src.marginx_;
  dst.marginy_=src.marginy_;

  for (int j=src.marginy_; j<src.rows_-src.marginy_; ++j){
    SrcPixel const* psrc = src.pcbits<SrcPixel>() + j*src.cols_ + src.marginx_;
    uint8_t* pdst = dst.data_ + j*dst.cols_ + src.marginx_;
    for (int i = src.marginx_; i<src.cols_-src.marginx_; ++i){
      *pdst++=clamp((*psrc - mymin) * factor);
      ++psrc;
    }
  }

  fillBorders(dst, clamp((SrcPixel(0) - mymin) * factor));
  return dst;
}

Image equalize(const Image& src, double mymin, double mymax){
  static const std::map<int, decltype(&equalizeWorker<int>)> 
      M={{1, &equalizeWorker<uint8_t>},{4, &equalizeWorker<float>}};
  return (*M.at(src.szPixel_))(src, mymin,mymax);
}

/////////////////////////////////////////////////////////////
// basic image IO

std::string print(const Image& img)
{
  //static std::map<int, decltype(&Image::at<int>)> M=
  //  {{1, &Image::at<uint8_t>},{4, &Image::at<float>}};
  //auto pmethod=M[img.szPixel_];
  //typedef int (core::Image::*MyMethod)(int, int) const;
  //MyMethod pmethod=0;
  //decltype(&Image::at<int>) pmethod=&Image::at<float>;

  std::ostringstream oss;
  oss <<std::setprecision(5);
  for (int row=0; row<img.rows_; ++row){
    for (int col=0; col<img.cols_; ++col){
      if (img.szPixel_==1){
        oss <<img.at<uint8_t>(row,col);
      } else if (img.szPixel_==4){
        oss <<img.at<float>(row,col);
      } else{
        assert(0);
      }
      oss <<"\t";
    }
    oss <<"\n";
  }
  return oss.str();
}

void savepgm(std::string filename, const Image& img)
{
  std::ofstream ofs(filename);
  ofs <<"P5\n" <<img.cols_ <<" " <<img.rows_ <<"\n" <<"255\n";
  ofs.write(img.pcbits<char>(), img.szBits());
}
Image loadpgm(std::string filename)
{
  std::string magic, rows, cols, maxpix;
  std::ifstream ifs(filename);
  ifs >>magic >>cols >>rows >>maxpix;
  assert((magic=="P5") && (maxpix=="255"));
  Image img(std::stoi(rows), std::stoi(cols), 1);
  ifs.read(img.pbits<char>(), img.szBits());
  return img;
}

//////////////////////////////////////////////////////////////////
// ImageSet

int ImageSet::szPixel()  const{
  const int sz=smooth_.szPixel_;
  assert(sz==gx_.szPixel_);
  assert(sz==gx_.szPixel_);
  return sz;
}
int ImageSet::rows()  const{
  const int rows=smooth_.rows_;
  assert(rows==gx_.rows_);
  assert(rows==gy_.rows_);
  return rows;
}
int ImageSet::cols()  const{
  const int cols=smooth_.cols_;
  assert(cols==gx_.cols_);
  assert(cols==gy_.cols_);
  return cols;
}
int ImageSet::szBits()  const{
  const int sz=smooth_.szBits();
  assert(sz==gx_.szBits());
  assert(sz==gx_.szBits());
  return sz;
}


//////////////////////////////////////////////////////////////////
// ImageSetExact

namespace{

std::vector<double> computeGaussKernel(
  double sigma, double k) 
{
  std::vector<double> kernel(2*int(k*sigma+0.5)+1);
  const int hw = kernel.size() / 2;

  // calculate values
  double sum=0;
  for (int i=-hw; i<=hw; ++i){
    double g=std::exp(-i*i / (2*sigma*sigma));
    kernel[i+hw] = g;
    sum+=g;
  }

  // normalize to 1
  for (auto& k_i: kernel){
    k_i /= sum;
  }
  return kernel;
}

std::vector<double> computeGaussDerivativeKernel(
  double sigma, double k) 
{
  std::vector<double> kernel(2*int(k*sigma+0.5)+1);
  const int hw = kernel.size() / 2;

  // calculate values
  double sum=0;
  for (int i=-hw; i<=hw; ++i){
    double g=std::exp(-i*i / (2*sigma*sigma));
    kernel[i+hw] = i*g;
    sum+=g;
  }

  // normalize...
  for (auto& k_i: kernel){
    k_i /= sum;
  }
  return kernel;
}
	

template <class SrcPixel>
void convolveImageHorizontal(
  const core::Image& src,
  const std::vector<double>& kernel,
  core::Image& dst)
{
  assert(src.szPixel_==sizeof(SrcPixel));
  assert(dst.szPixel_==4);

  int rows=src.rows_;
  int cols=src.cols_;
  assert(cols == dst.cols_);
  assert(rows == dst.rows_);
  int marginadd = kernel.size() / 2;
  assert(kernel.size() == 2*marginadd+1);
  dst.marginx_ = src.marginx_ + marginadd;
  dst.marginy_ = src.marginy_;

  int srcStride=-kernel.size()+1;
  for (int j=dst.marginy_; j<rows-dst.marginy_; ++j){
    SrcPixel const* psrc = src.pcbits<SrcPixel>() + j*cols + src.marginx_;
    float*   pdst = dst.pbits<float>() + j*cols + dst.marginx_;
    for (int i = dst.marginx_; i<cols-dst.marginx_; ++i){
      double sum = 0.0;
      for (int k = 0; k<kernel.size(); ++k){
        sum += *psrc++ * kernel[k];
      }
      *pdst++ = sum;
      psrc += srcStride;
    }
  }
}

template <class SrcPixel>
void convolveImageVertical(
  const core::Image& src,
  const std::vector<double>& kernel,
  core::Image& dst)
{
  assert(src.szPixel_==sizeof(SrcPixel));
  assert(dst.szPixel_==4);

  int rows=src.rows_;
  int cols=src.cols_;
  assert(cols == dst.cols_);
  assert(rows == dst.rows_);
  int marginadd = kernel.size() / 2;
  assert(kernel.size() == 2*marginadd+1);
  dst.marginx_ = src.marginx_;
  dst.marginy_ = src.marginy_ + marginadd;

  int srcStride=-kernel.size()*cols+1;
  for (int j=dst.marginy_; j<rows-dst.marginy_; ++j){
    SrcPixel const* psrc = src.pcbits<SrcPixel>() + (j-marginadd)*cols + dst.marginx_;
    float*   pdst = dst.pbits<float>() + j*cols + dst.marginx_;
    for (int i = dst.marginx_; i<cols-dst.marginx_; ++i){
      double sum = 0.0;
      for (int k = 0; k<kernel.size(); ++k){
        sum += *psrc * kernel[k];
        // skip to next row
        psrc += cols;
      }
      *pdst++ = sum;
      psrc += srcStride;
    }
  }
}


template <class SrcPixel>
void convolveSeparate(
  const core::Image& src,
  const std::vector<double>& kernelx,
  const std::vector<double>& kernely,
  core::Image& dst)
{
  core::Image tmp(src.rows_, src.cols_, 4);
  convolveImageHorizontal<SrcPixel>(src, kernelx, tmp);
  convolveImageVertical<float>(tmp, kernely, dst);
}


} // unnamed namespace ends here

ImageSetExact& ImageSetExact::operator=(ImageSetExact& other)
{
  this->smooth_ = other.smooth_;
  this->gx_ = other.gx_;
  this->gy_ = other.gy_;
  this->kernelSmooth_ = other.kernelSmooth_;
  this->kernelGrad_ = other.kernelGrad_;
  return *this;
}

void ImageSetExact::compute(const Image& src){
  smooth_.resize(src.rows_, src.cols_, 4);
  std::vector<double> kg  = computeGaussKernel(sigmaSmoothing_, kSigma_);
  if (src.szPixel_==1){
    convolveSeparate<uint8_t>(src, kg,kg, smooth_);
  } else{
    convolveSeparate<float>(src, kg,kg, smooth_);
  }

  gx_.resize(src.rows_, src.cols_, 4);
  gy_.resize(src.rows_, src.cols_, 4);
  std::vector<double> kgd = computeGaussDerivativeKernel(sigmaGradient_, kSigma_);
  std::vector<double> kg2  = computeGaussKernel(sigmaGradient_, kSigma_);
  convolveSeparate<float>(smooth_, kgd, kg2, gx_);
  convolveSeparate<float>(smooth_, kg2, kgd, gy_);

  fillBorders(smooth_, float(0));
  fillBorders(gx_, float(0));
  fillBorders(gy_, float(0));
}


void ImageSetExact::config(const std::string& conf){

}

/////////////////////////////////////////////////////////////
// test images

Image maketest(int rows, int cols, int delta){
  int white=200;
  int black=100;
  Image img(rows, cols, 1);
  for (int j=0; j<img.rows_; ++j){
    uint8_t* pimg = img.data_ + j*img.cols_;
    for (int i = 0; i<img.cols_; ++i){
      bool is_white=int(i/delta)%2 == int(j/delta)%2;
      *pimg++=black+(white-black)*is_white;
    }
  }
  return img;
}

//////////////////////////////////////////////////////////////////
// tests

namespace{
   int test(){
      Image img;
      Image img2(img);
      Image img3;
      img3=img;     
      return 0;
   }
   //static int bla=test();
}

}
