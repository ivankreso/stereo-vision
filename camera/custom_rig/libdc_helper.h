#ifndef CAMERA_LIBDC_HELPER_
#define CAMERA_LIBDC_HELPER_

//#include <opencv2/core/core.hpp>
#include <dc1394/dc1394.h>

namespace cam {

namespace LibdcHelper
{
  void PrintVideoMode(uint32_t mode);
  void PrintTriggerMode(uint32_t mode);
  void PrintTriggerSource(uint32_t mode);
  void PrintColorCoding(uint32_t coding);  
  void PrintColorFilter(dc1394color_filter_t& mode);  
}

#define print_case(A) case A: printf(#A ""); break;

inline
void LibdcHelper::PrintVideoMode(uint32_t mode)
{
  switch (mode) {
    print_case(DC1394_VIDEO_MODE_160x120_YUV444);
    print_case(DC1394_VIDEO_MODE_320x240_YUV422);
    print_case(DC1394_VIDEO_MODE_640x480_YUV411);
    print_case(DC1394_VIDEO_MODE_640x480_YUV422);
    print_case(DC1394_VIDEO_MODE_640x480_RGB8);
    print_case(DC1394_VIDEO_MODE_640x480_MONO8);
    print_case(DC1394_VIDEO_MODE_640x480_MONO16);
    print_case(DC1394_VIDEO_MODE_800x600_YUV422);
    print_case(DC1394_VIDEO_MODE_800x600_RGB8);
    print_case(DC1394_VIDEO_MODE_800x600_MONO8);
    print_case(DC1394_VIDEO_MODE_1024x768_YUV422);
    print_case(DC1394_VIDEO_MODE_1024x768_RGB8);
    print_case(DC1394_VIDEO_MODE_1024x768_MONO8);
    print_case(DC1394_VIDEO_MODE_800x600_MONO16);
    print_case(DC1394_VIDEO_MODE_1024x768_MONO16);
    print_case(DC1394_VIDEO_MODE_1280x960_YUV422);
    print_case(DC1394_VIDEO_MODE_1280x960_RGB8);
    print_case(DC1394_VIDEO_MODE_1280x960_MONO8);
    print_case(DC1394_VIDEO_MODE_1600x1200_YUV422);
    print_case(DC1394_VIDEO_MODE_1600x1200_RGB8);
    print_case(DC1394_VIDEO_MODE_1600x1200_MONO8);
    print_case(DC1394_VIDEO_MODE_1280x960_MONO16);
    print_case(DC1394_VIDEO_MODE_1600x1200_MONO16);
    print_case(DC1394_VIDEO_MODE_EXIF);
    print_case(DC1394_VIDEO_MODE_FORMAT7_0);
    print_case(DC1394_VIDEO_MODE_FORMAT7_1);
    print_case(DC1394_VIDEO_MODE_FORMAT7_2);
    print_case(DC1394_VIDEO_MODE_FORMAT7_3);
    print_case(DC1394_VIDEO_MODE_FORMAT7_4);
    print_case(DC1394_VIDEO_MODE_FORMAT7_5);
    print_case(DC1394_VIDEO_MODE_FORMAT7_6);
    print_case(DC1394_VIDEO_MODE_FORMAT7_7);

  default:
    dc1394_log_error("Unknown format\n");
    throw 1;
  }
}

inline
void LibdcHelper::PrintTriggerSource(uint32_t mode)
{
  std::cout << mode << "\n";
  switch (mode) {
    print_case(DC1394_TRIGGER_SOURCE_0);
    print_case(DC1394_TRIGGER_SOURCE_1);
    print_case(DC1394_TRIGGER_SOURCE_2);
    print_case(DC1394_TRIGGER_SOURCE_3);
    print_case(DC1394_TRIGGER_SOURCE_SOFTWARE);
  default:
    dc1394_log_error("Unknown format\n");
    throw 1;
  }
}

inline
void LibdcHelper::PrintColorCoding(uint32_t coding)
{
  std::cout << coding << "\n";
  switch (coding) {
    print_case(DC1394_COLOR_CODING_MONO8);
    print_case(DC1394_COLOR_CODING_YUV411);
    print_case(DC1394_COLOR_CODING_YUV422);
    print_case(DC1394_COLOR_CODING_YUV444);
    print_case(DC1394_COLOR_CODING_RGB8);
    print_case(DC1394_COLOR_CODING_MONO16);
    print_case(DC1394_COLOR_CODING_RGB16);
    print_case(DC1394_COLOR_CODING_MONO16S);
    print_case(DC1394_COLOR_CODING_RGB16S);
    print_case(DC1394_COLOR_CODING_RAW8);
    print_case(DC1394_COLOR_CODING_RAW16);
  default:
    dc1394_log_error("Unknown format\n");
    throw 1;
  }
}

inline
void LibdcHelper::PrintTriggerMode(uint32_t mode)
{
  switch (mode) {
    print_case(DC1394_TRIGGER_MODE_0);
    print_case(DC1394_TRIGGER_MODE_1);
    print_case(DC1394_TRIGGER_MODE_2);
    print_case(DC1394_TRIGGER_MODE_3);
    print_case(DC1394_TRIGGER_MODE_4);
    print_case(DC1394_TRIGGER_MODE_5);
    print_case(DC1394_TRIGGER_MODE_14);
    print_case(DC1394_TRIGGER_MODE_15);

  default:
    dc1394_log_error("Unknown format\n");
    throw 1;
  }
}

inline
void LibdcHelper::PrintColorFilter(dc1394color_filter_t& mode)
{
  switch (mode) {
    print_case(DC1394_COLOR_FILTER_RGGB);
    print_case(DC1394_COLOR_FILTER_GBRG);
    print_case(DC1394_COLOR_FILTER_GRBG);
    print_case(DC1394_COLOR_FILTER_BGGR);

  default:
    dc1394_log_error("Unknown format\n");
    throw 1;
  }
}

//typedef enum {
//    DC1394_FEATURE_BRIGHTNESS= 416,
//    DC1394_FEATURE_EXPOSURE,
//    DC1394_FEATURE_SHARPNESS,
//    DC1394_FEATURE_WHITE_BALANCE,
//    DC1394_FEATURE_HUE,
//    DC1394_FEATURE_SATURATION,
//    DC1394_FEATURE_GAMMA,
//    DC1394_FEATURE_SHUTTER,
//    DC1394_FEATURE_GAIN,
//    DC1394_FEATURE_IRIS,
//    DC1394_FEATURE_FOCUS,
//    DC1394_FEATURE_TEMPERATURE,
//    DC1394_FEATURE_TRIGGER,
//    DC1394_FEATURE_TRIGGER_DELAY,
//    DC1394_FEATURE_WHITE_SHADING,
//    DC1394_FEATURE_FRAME_RATE,
//    DC1394_FEATURE_ZOOM,
//    DC1394_FEATURE_PAN,
//    DC1394_FEATURE_TILT,
//    DC1394_FEATURE_OPTICAL_FILTER,
//    DC1394_FEATURE_CAPTURE_SIZE,
//    DC1394_FEATURE_CAPTURE_QUALITY
//} dc1394feature_t;

//typedef enum {
//    DC1394_FEATURE_MODE_MANUAL= 736,
//    DC1394_FEATURE_MODE_AUTO,
//    DC1394_FEATURE_MODE_ONE_PUSH_AUTO
//} dc1394feature_mode_t;

}

#endif
