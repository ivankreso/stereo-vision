#include "feature_refiner_klt.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>

#include <cassert>

namespace{

/////////////////////////////////////////////////////////////
// image processing

double interpolate(
  double x, double y,
  const core::Image& img)
{
  assert(img.szPixel_==4);
  const int x_floor = (int)x;
  const int y_floor = (int)y;
  assert(x_floor>=0 && x_floor<img.cols_-1);
  assert(y_floor>=0 && y_floor<img.rows_-1);

  float const *psrc = (float const *) img.data_;
  float const *ppix0 = psrc + y_floor * img.cols_ + x_floor;
  float const *ppix1 = ppix0 + img.cols_;

  double dx = x - x_floor;
  double dy = y - y_floor;
  double dxdy = dx * dy;
  double result = dxdy * ppix1[1];
  result += (dx - dxdy) * ppix0[1];
  result += (dy - dxdy) * ppix1[0];
  result += (1 -dx -dy +dxdy) * ppix0[0];
  return result;
}

void getReferenceImages(
  const core::Image& src, /* source image */
  track::refiner::FeatureData& fi) /* feature position, output */
{
  assert(src.szPixel_==4);
  const int hw=fi.width()/2;
  const int hh=fi.height()/2;
  float* pmagdst=(float*)fi.ref_.data_;
  for (int y=-hh; y<=hh; ++y){
    for (int x=-hw ; x<=hw; ++x)
      *pmagdst++ = interpolate(fi.posx() + x, fi.posy() + y, src);
  }
}

// old without subpixel references
//void getReferenceImages(
//  const core::Image& src, /* source image */
//  track::refiner::FeatureData& fi) /* feature position, output */
//{
//  assert(src.szPixel_==4);
//  const int x0 = std::round(fi.posx());
//  assert(fi.posx() == x0);
//  const int y0 = std::round(fi.posy());
//  assert(fi.posy() == y0);

//  const int hw=fi.width()/2;
//  assert(x0>hw && x0<src.cols_-hw);
//  const int hh=fi.height()/2;
//  assert(y0>hh && y0<src.rows_-hh);
//  float* pmagdst=(float*)fi.ref_.data_;
// 
//  for (int y=y0-hh; y<=y0+hh; ++y){
//    float const* pmagsrc=src.pcbits<float>() + y*src.cols_ + x0-hw;
//    float const* pmagsrcEnd=pmagsrc+2*hw+1;
//    while (pmagsrc<pmagsrcEnd){
//      *pmagdst++=*pmagsrc++;
//    }
//  }
//}

void computeWarpedGradient(
  const core::ImageSet& src,   /* source images */
  const track::refiner::FeatureData& fi,     /* feature position */
  const core::Image& gwgx,     /* geometrically warped gradient x */
  const core::Image& gwgy)     /* geometrically warped gradient y */
{
  float* pgwgx =(float*)gwgx.data_;
  float* pgwgy =(float*)gwgy.data_;

  const int hw=fi.width()/2;
  const int hh=fi.height()/2;
  for (int y=-hh; y<=hh; ++y){
    for (int x=-hw ; x<=hw; ++x){
      double xsrc = fi.warpx(x,y);
      double ysrc = fi.warpy(x,y);
      *pgwgx++ = interpolate(xsrc,ysrc, src.gx_);
      *pgwgy++ = interpolate(xsrc,ysrc, src.gy_);
    }
  }
}

double computeError(
  const core::ImageSet& src,   /* source images */
  track::refiner::FeatureData& fi,           /* feature position, some outputs */
  core::Image& gwimg)    /* geometrically warped image */
{
  double residue=0;
  float const* pref =(float const*)fi.ref_.data_; // reference
  float* pgwimg     =(float*)gwimg.data_;         // geometric warp
  float* pwarped    =(float*)fi.warped_.data_;    // geometric+photometric warp
  float* perror     =(float*)fi.error_.data_;

  const int hw=fi.width()/2;
  const int hh=fi.height()/2;
  for (int y=-hh; y<=hh; ++y){
    for (int x=-hw ; x<=hw; ++x){
      double gw_smooth=interpolate(fi.warpx(x,y),fi.warpy(x,y), src.smooth_);
      double warped = fi.lambda() * gw_smooth + fi.delta();
      double error = warped - *pref++;
      *pgwimg++ = gw_smooth;
      *pwarped++ = warped;
      *perror++ = error;
      residue+= error*error;
    }
  }
  return sqrt(residue/fi.width()/fi.height());
}



/////////////////////////////////////////////////////////////
// linear system solving

// taken (almost) verbatim from http://www.ces.clemson.edu/~stb/klt/
int _am_gauss_jordan_elimination(double **a, int n, double **b, int m)
{
  /* re-implemented from Numerical Recipes in C */
  int *indxc,*indxr,*ipiv;
  int i,j,k,l,ll;
  double big,dum,pivinv;
  int col = 0;
  int row = 0;

  assert(n > 0);
  indxc=(int *)malloc((size_t) (n*sizeof(int)));
  indxr=(int *)malloc((size_t) (n*sizeof(int)));
  ipiv=(int *)malloc((size_t) (n*sizeof(int)));
  for (j=0;j<n;j++) ipiv[j]=0;
  for (i=0;i<n;i++) {
    big=0.0;
    for (j=0;j<n;j++)
      if (ipiv[j] != 1)
        for (k=0;k<n;k++) {
          if (ipiv[k] == 0) {
            if (fabs(a[j][k]) >= big) {
              big=fabs(a[j][k]);
              row=j;
              col=k;
            }
          } else if (ipiv[k] > 1) return false;
        }
    ++(ipiv[col]);
    if (row != col) {
      for (l=0;l<n;l++) std::swap(a[row][l],a[col][l]);
      for (l=0;l<m;l++) std::swap(b[row][l],b[col][l]);
    }
    indxr[i]=row;
    indxc[i]=col;
    if (a[col][col] == 0.0) return false;
    pivinv=1.0f/a[col][col];
    a[col][col]=1.0;
    for (l=0;l<n;l++) a[col][l] *= pivinv;
    for (l=0;l<m;l++) b[col][l] *= pivinv;
    for (ll=0;ll<n;ll++)
      if (ll != col) {
        dum=a[ll][col];
        a[ll][col]=0.0;
        for (l=0;l<n;l++) a[ll][l] -= a[col][l]*dum;
        for (l=0;l<m;l++) b[ll][l] -= b[col][l]*dum;
      }
  }
  for (l=n-1;l>=0;l--) {
    if (indxr[l] != indxc[l])
      for (k=0;k<n;k++)
        std::swap(a[k][indxr[l]],a[k][indxc[l]]);
  }
  free(ipiv);
  free(indxr);
  free(indxc);

  return true;
}

std::vector<double> solve(
  std::vector<std::vector<double> >& LS)
{
  int n=LS.size()-1;
  double** a=(double**)malloc(n * sizeof(double*));
  double** b=(double**)malloc(n * sizeof(double*));
  
  for (int i=0; i<n; ++i){
    a[i]=&LS[i][0];
    b[i]=&LS[n][i];
  }  
  bool success=_am_gauss_jordan_elimination(a,n,b,1);

  std::vector<double> result;
  if (success){
    result.resize(n);
    for (int i=0; i<n; ++i){
      result[i] = -b[i][0];   // Ax+b=0 vs Ax=b
    }  
  }

  free(a); free(b);

  return result;
}

int testSolver(){
  try{throw 1;} catch (int){};
  std::vector<std::vector<double> > LS(3);
  LS[0]={1,2};
  LS[1]={2,2};
  LS[2]={5,6};
  std::vector<double> result=solve(LS);
  std::cerr <<"solution: ";
  for (auto q: result){
    std::cerr <<q <<" ";
  }
  std::cerr <<"\n";
  exit(0);
  return 0;
}
//bool success=testSolver();



/////////////////////////////////////////////////////////////////////////////////
// Various trackers


class LKTracker{
protected:  
  int hw_;
  int hh_;
  std::vector<std::vector<double> > LS_;
  std::vector<double> solution_;
  std::vector<double> g_;
public:
  virtual std::vector<double> track(
    float const* pgwgx, float const* pgwgy, float const* perror, 
    float const* pgwimg, double lambda);
protected:
  virtual void constructSystem(
    float const* pgwgx, float const* pgwgy, float const* perror, 
    float const* pgwimg, double lambda)=0;
  virtual void adjustSolution(){}
protected:
  void updateLinearSystem(double diff);
};  

void LKTracker::updateLinearSystem(double diff)
{
  for (size_t i=0; i<g_.size(); ++i){
    for (size_t j=i+1; j<g_.size(); ++j){
      LS_[i][j]+=g_[i]*g_[j];
      LS_[j][i]+=g_[i]*g_[j];
    }
    LS_[i][i]+=g_[i]*g_[i];
    LS_[g_.size()][i]+=diff*g_[i];
  }
}

std::vector<double> LKTracker::track(
  float const* pgwgx, float const* pgwgy, float const* perror, 
  float const* pgwimg, double lambda)
{
  constructSystem(pgwgx, pgwgy, perror, pgwimg, lambda);
  solution_=solve(LS_);
  if (solution_.size()!=0){
    adjustSolution();
  }
  return solution_;
}

class LKTracker2: public LKTracker{
public:
  LKTracker2(int hw, int hh){hw_=hw; hh_=hh;}
  virtual void constructSystem(
    float const* gwgx, float const* gwgy, float const* imgdiff, 
    float const* gwimg, double lambda);
};  
void LKTracker2::constructSystem(
  float const* gwgx, float const* gwgy, float const* imgdiff, 
  float const* , double )
{
  LS_.assign(3, std::vector<double>(2,0));  
  g_.resize(2);  
  for (int y=-hh_; y<=hh_; ++y){
    for (int x=-hw_; x<=hw_; ++x){
      g_[0]=*gwgx++;
      g_[1]=*gwgy++;
      updateLinearSystem(*imgdiff++);      
    }
  }
}

class LKTracker5: public LKTracker{
public:
  LKTracker5(int hw, int hh){hw_=hw; hh_=hh;}
protected:
  virtual void constructSystem(
    float const* gwgx, float const* gwgy, float const* imgdiff, 
    float const* gwimg, double lambda);
  virtual void adjustSolution();
};  
void LKTracker5::constructSystem(
  float const* gwgx, float const* gwgy, float const* imgdiff, 
  float const* gwimg, double lambda)
{
  LS_.assign(6, std::vector<double>(5,0));  
  g_.resize(5);
  for (int y=-hh_; y<=hh_; ++y){
    for (int x=-hw_; x<=hw_; ++x){
      double gx = *gwgx++;
      double gy = *gwgy++;
      double diff = *imgdiff++;
      double i2 = *gwimg++;
      g_[0]=lambda*gx;
      g_[1]=lambda*gy;
      g_[2]=lambda*(x*gx+y*gy);
      g_[3]=i2;
      g_[4]=1;
      updateLinearSystem(diff);      
    }
  }
}
void LKTracker5::adjustSolution(){
  assert(solution_.size()==5);
  solution_.resize(8);
  solution_[7]=solution_[4];
  solution_[6]=solution_[3];
  solution_[5]=solution_[2];
  solution_[4]=solution_[3]=0;
}

class LKTracker8: public LKTracker{
public:
  LKTracker8(int hw, int hh){hw_=hw; hh_=hh;}
protected:
  virtual void constructSystem(
    float const* gwgx, float const* gwgy, float const* imgdiff, 
    float const* gwimg, double lambda);
};  
void LKTracker8::constructSystem(
  float const* gwgx, float const* gwgy, float const* imgdiff, 
  float const* gwimg, double lambda)
{
  LS_.assign(9, std::vector<double>(8,0));
  g_.resize(8);
  for (int y=-hh_; y<=hh_; ++y){
    for (int x=-hw_; x<=hw_; ++x){
      double gx = *gwgx++;
      double gy = *gwgy++;
      double i2 = *gwimg++;
      double diff = *imgdiff++;
      g_[0]=lambda*gx;
      g_[1]=lambda*gy;
      g_[2]=lambda*gx*x;
      g_[3]=lambda*gx*y;
      g_[4]=lambda*gy*x;
      g_[5]=lambda*gy*y;
      g_[6]=i2;
      g_[7]=1;
      updateLinearSystem(diff);      
    }
  }
}

LKTracker* createLKTracker(int hw, int hh, int model){
  switch (model){
  case 2:
    return new LKTracker2(hw,hh);
  case 5:
    return new LKTracker5(hw,hh);
  case 8:
    return new LKTracker8(hw,hh);
  }
  assert(0);
  return nullptr;
}
/////////////////////////////////////////
// convergence tests

bool testOutOfBoundary(
  const std::vector<core::Point>& bbox, 
  int rows, int cols)
{
  bool rv=false;
  for (const auto& bbox_i: bbox){
    rv |= bbox_i.x_ < 0;
    rv |= bbox_i.x_ > cols-1;
    rv |= bbox_i.y_ < 0;
    rv |= bbox_i.y_ > rows-1;
  }
  return rv;
}

std::vector<core::Point> subtract(
  const std::vector<core::Point>& bbox1, 
  const std::vector<core::Point>& bbox2)
{
  std::vector<core::Point> diff(bbox1.size());
  for (size_t i=0; i<bbox1.size(); ++i){
    diff[i]=bbox1[i]-bbox2[i];
  }
  return diff;
}

bool testConvergence(
  const std::vector<core::Point>& bboxCur, 
  const std::vector<core::Point>& bboxPrev,
  double thConvergence)
{
  std::vector<core::Point> diff(subtract(bboxCur, bboxPrev));

  bool rv=true;
  for (const auto& diff_i: diff){
    rv &= diff_i.l1() < thConvergence;
  }
  return rv;
}
bool testDivergence(
  const std::vector<core::Point>& bboxCur, 
  const std::vector<core::Point>& bboxInitial,
  int thDivergence)
{
  std::vector<core::Point> diff(subtract(bboxCur, bboxInitial));

  bool rv=false;
  for (const auto& diff_i: diff){
    rv |= diff_i.l1() >= thDivergence;
  }
  return rv;
}

/////////////////////////////////////////
// reporting

void reportIteration(int fid, int iteration, 
  const track::refiner::FeatureData& feature, 
  const std::vector<double>& solution)
{
  try{throw 1;} catch (int){}; // catch breakpoint
 
  std::ostringstream oss;
  oss <<"F" <<fid <<"#" <<std::setfill('0') <<std::setw(3) <<iteration;
  std::cerr <<oss.str() <<"@" <<feature.pt() <<"R" <<feature.residue_ <<"\n";

  //core::savepgm(oss.str()+"gwgx.pgm", core::equalize(gwgx));
  //track::refiner::savepgm(oss.str()+"gwgy.pgm", track::refiner::equalize(gwgy));
  //track::refiner::savepgm(oss.str()+"gwimg.pgm", track::refiner::equalize(gwimg));
  core::savepgm(oss.str()+"ref.pgm", core::equalize(feature.ref_, 0,255));
  core::savepgm(oss.str()+"warped.pgm", core::equalize(feature.warped_, 0,255));
  core::savepgm(oss.str()+"error.pgm", core::equalize(feature.error_,-255,255));

  std::ostringstream ossw;
  ossw <<"warp: ";
  for (int i=0; i<8; ++i){
    ossw <<feature.warp_[i] <<" ";
  }
  std::cerr <<"  " <<ossw.str() <<"\n";

  std::ostringstream ossdw;
  ossdw <<"delta warp: ";
  for (auto dw: solution){
    ossdw <<dw <<" ";
  }
  std::cerr <<"  " <<ossdw.str() <<"\n";
}

} // unnamed namespace ends here


//////////////////////////////////////////
// The interface class: FeatureRefinerKLT
namespace track {
namespace refiner {

void FeatureRefinerKLT::config(const std::string& conf)
{
}

void FeatureRefinerKLT::addFeatures(
  const core::ImageSet& src,
  const std::map<int, core::Point>& pts)
{
  // TODO
  //#pragma omp parallel for
  for (auto q : pts){
  //for (auto iter = pts.begin(); iter != pts.end(); iter++) {
    // TODO - replace or not to replace?
    //auto ret = map_.emplace(q.first, FeatureData(q.second.x_, q.second.y_));
    map_[q.first] = FeatureData(q.second.x_, q.second.y_);
    //auto ret = map_.emplace(iter->first, FeatureData(iter->second.x_, iter->second.y_));
    //std::cout << "index: " << q.first << "\n";
    //assert(ret.second == true);

    FeatureData& feature = map_[q.first];
    //FeatureData& feature = map_[iter->first];

    //std::cout << feature.pt() << "\n";
    getReferenceImages(src.smooth_, feature);
  }
}

void FeatureRefinerKLT::refineFeatures(
  const core::ImageSet& src,
  const std::map<int, core::Point>& pts)
{
  if (verbose_){
    core::savepgm("srcgx.pgm", core::equalize(src.gx_));
    core::savepgm("srcgy.pgm", core::equalize(src.gy_));
    core::savepgm("srcimg.pgm", core::equalize(src.smooth_, 0, 255));
  }

  // local images
  const int fw=FeatureData::width();
  const int fh=FeatureData::height();
  core::Image gwgx(fw,fh, 4);  // geometrically warped gradient x
  core::Image gwgy(fw,fh, 4);  // geometrically warped gradient y
  core::Image gwimg(fw,fh, 4); // geometrically warped image

  std::unique_ptr<LKTracker> plk(createLKTracker(fw/2,fh/2, warpModel_));

  // TODO this was removed coz of xeon phi gcc 4.7
  //#pragma omp parallel for
  for (auto q : pts){
    FeatureData& feature=map_[q.first];
    feature.setpos(q.second.x_,q.second.y_);

    // check bounding box
    std::vector<core::Point> bboxInitial=feature.bbox();
    if (testOutOfBoundary(bboxInitial, src.rows(),src.cols())){
      feature.status_=FeatureData::OutOfBounds;
      continue;
    }

    // iterate until convergence or failure
    feature.status_=FeatureData::MaxIterations;
    std::vector<core::Point> bboxCur(bboxInitial);
    for (int iteration=0; iteration<thMaxIterations_; ++iteration){
      // compute error
      computeWarpedGradient(src, feature, gwgx, gwgy);
      feature.residue_=computeError(src, feature, gwimg);
      if(iteration == 0)
        feature.first_residue_ = feature.residue_;


      // LK tracking
      std::vector<double> improvement = plk->track(
         gwgx.pcbits<float>(),
         gwgy.pcbits<float>(),
         feature.error_.pcbits<float>(),
         gwimg.pcbits<float>(), feature.lambda());
      if (improvement.size()==0){
        feature.status_=FeatureData::SmallDet;
        break;
      }
      if (verbose_){
        reportIteration(q.first, iteration, feature, improvement);
      }
      for (size_t i=0; i<improvement.size(); ++i){
        feature.warp_[i]+=improvement[i]*optimizationFactor_;
      }

      // check bounding box
      std::vector<core::Point> bboxPrev(std::move(bboxCur));
      bboxCur=feature.bbox();
      if (testOutOfBoundary(bboxCur, src.rows(),src.cols())){
        feature.status_=FeatureData::OutOfBounds;
        break;
      }

      // check convergence
      if (testConvergence(bboxCur, bboxPrev, thDisplacementConvergence_) &&
          feature.residue_ < thMaxResidue_)
      {
        feature.status_ = FeatureData::OK;
        break;
      }
    }

    if (feature.status_ == FeatureData::OK){
      // check divergence
      if (testDivergence(bboxCur, bboxInitial, thDisplacementDivergence_)){
        feature.status_=FeatureData::NotFound;
      }
      feature.residue_=computeError(src, feature, gwimg);
      // check residue
      if (feature.residue_>=thMaxResidue_){
        feature.status_=FeatureData::LargeResidue;
      }
    }
    // TODO
    // check if the final residue is worse then initial
    //if (feature.status_ == FeatureData::OK){
    //  if(feature.residue_ > feature.first_residue_)
    //    std::cout << "[FeatureRefinerKLT] First residue -> New residue: " << feature.first_residue_
    //              << " -> " << feature.residue_ << "\n";
    //}

    if (verbose_){
      std::ostringstream oss;
      oss <<"F" <<q.first <<"S" <<feature.status_
          <<"@" <<feature.pt() <<"R" <<feature.residue_ <<"\n";
      std::cerr <<oss.str() <<"\n";
    }
  }
  return;
}

}
}

