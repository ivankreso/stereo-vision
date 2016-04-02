#include <opencv2/core/core.hpp>


namespace recon
{

enum { MODE_SGBM = 0,
       MODE_HH   = 1
     };

struct StereoSGBMParams
{
  StereoSGBMParams()
  {
    minDisparity = numDisparities = 0;
    SADWindowSize = 0;
    P1 = P2 = 0;
    disp12MaxDiff = 0;
    preFilterCap = 0;
    uniquenessRatio = 0;
    speckleWindowSize = 0;
    speckleRange = 0;
    mode = MODE_HH;
  }

  StereoSGBMParams(int _minDisparity, int _numDisparities, int _SADWindowSize,
      int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
      int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
      int _mode)
  {
    minDisparity = _minDisparity;
    numDisparities = _numDisparities;
    SADWindowSize = _SADWindowSize;
    P1 = _P1;
    P2 = _P2;
    disp12MaxDiff = _disp12MaxDiff;
    preFilterCap = _preFilterCap;
    uniquenessRatio = _uniquenessRatio;
    speckleWindowSize = _speckleWindowSize;
    speckleRange = _speckleRange;
    mode = _mode;
  }

  int minDisparity;
  int numDisparities;
  int SADWindowSize;
  int preFilterCap;
  int uniquenessRatio;
  int P1;
  int P2;
  int speckleWindowSize;
  int speckleRange;
  int disp12MaxDiff;
  int mode;
};

void computeDisparitySGBM(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& disp1,
                          const StereoSGBMParams& params, cv::Mat& buffer);

void computeSGM(cv::Mat& left, cv::Mat& right, StereoSGBMParams& params, cv::Mat& disp);

}
