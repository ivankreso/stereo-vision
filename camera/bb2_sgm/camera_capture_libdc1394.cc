#include "camera_capture_libdc1394.h"

#include <iostream>
#include <fstream>

#include "libdc_helper.h"

namespace {

// Releases the cameras and exits
void CleanupAndExit(dc1394camera_t *camera)
{
  dc1394_video_set_transmission(camera, DC1394_OFF);
  dc1394_capture_stop(camera);
  dc1394_camera_free(camera);
  exit(1);
}

}

namespace cam {

CameraCaptureLibdc1394::CameraCaptureLibdc1394(uint64_t cam_guid, dc1394_t* bus_data, bool use_external_trigger) :
    cam_guid_(cam_guid), ring_buffer_size_(500), use_external_trigger_(use_external_trigger) {
  // 3 - the best - no frame drops
  frames_.assign(ring_buffer_size_, nullptr);
  camera_ = dc1394_camera_new(bus_data, cam_guid);
  if (!camera_) {
    dc1394_log_error("Failed to initialize camera with guid %llx", cam_guid);
    return;
  }
  std::cout << "Using camera with GUID = " << cam_guid << "\n";

  dc1394video_modes_t modes;
  /*-----------------------------------------------------------------------
   *  list Capture Modes
   *-----------------------------------------------------------------------*/
  dc1394error_t err = dc1394_video_get_supported_modes(camera_, &modes);
  if (err != 0) {
    std::cout << "Could not get list of modes.\n";
    return;
  }
  std::cout << "Video modes:\n";
  for (uint32_t i = 0; i < modes.num; i++) {
    std::cout << "Mode [" << i << "] = ";
    LibdcHelper::PrintVideoMode(modes.modes[i]);
    std::cout << "\n";
  }

  dc1394trigger_sources_t trigger_modes;
  dc1394_external_trigger_get_supported_sources(camera_, &trigger_modes);
  std::cout << "\nTrigger sources:\n";
  for (uint32_t i = 0; i < trigger_modes.num; i++) {
    std::cout << "Mode [" << i << "] = ";
    LibdcHelper::PrintTriggerSource(trigger_modes.sources[i]);
    std::cout << "\n";
  }
  std::cout << "\n";
  // select last mode - usually camera specific FORMAT_7
  video_mode_ = modes.modes[modes.num-1];
  //video_mode_ = DC1394_VIDEO_MODE_FORMAT7_0;
  //video_mode_ = modes.modes[2];

  std::cout << "\n";
  dc1394color_codings_t color_codings;
  dc1394_format7_get_color_codings(camera_, video_mode_, &color_codings);
  for (uint32_t i = 0; i < color_codings.num; i++) {
    std::cout << "Mode [" << i+1 << "/" << color_codings.num << "] = ";
    LibdcHelper::PrintColorCoding(color_codings.codings[i]);
    std::cout << "\n";
  }
  //dc1394_format7_set_color_coding(camera_, video_mode_, DC1394_COLOR_CODING_RAW16);
  dc1394_format7_set_color_coding(camera_, video_mode_, DC1394_COLOR_CODING_MONO16);

  dc1394_video_set_operation_mode(camera_, DC1394_OPERATION_MODE_LEGACY);
  // setup capture
  err = dc1394_video_set_iso_speed(camera_, DC1394_ISO_SPEED_400);
  DC1394_ERR_CLN(err, CleanupAndExit(camera_), "Could not set iso speed\n");

  err = dc1394_video_set_mode(camera_, video_mode_);
  DC1394_ERR_CLN(err, CleanupAndExit(camera_), "Could not set video mode\n");

  //err = dc1394_video_set_framerate(camera_, DC1394_FRAMERATE_15);
  //err = dc1394_video_set_framerate(camera_, DC1394_FRAMERATE_30);
  //err = dc1394_video_set_framerate(camera_, DC1394_FRAMERATE_7_5);
  DC1394_ERR_CLN(err, CleanupAndExit(camera_), "Could not set framerate\n");

  // exposure and brightness are only for auto-shutter
  //err = dc1394_feature_set_value(camera_, DC1394_FEATURE_EXPOSURE, 50);
  //err = dc1394_feature_set_mode(camera_, DC1394_FEATURE_BRIGHTNESS, DC1394_FEATURE_MODE_AUTO);
  //err = dc1394_feature_set_mode(camera_, DC1394_FEATURE_EXPOSURE, DC1394_FEATURE_MODE_AUTO);
  //err = dc1394_feature_set_mode(camera_, DC1394_FEATURE_WHITE_BALANCE, DC1394_FEATURE_MODE_AUTO);
  //err = dc1394_feature_set_mode(camera_, DC1394_FEATURE_GAMMA, DC1394_FEATURE_MODE_AUTO);
  //err = dc1394_feature_set_mode(camera_, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_AUTO);


  //err = dc1394_format7_set_roi(camera_, DC1394_VIDEO_MODE_FORMAT7_3, DC1394_COLOR_CODING_RAW16, DC1394_USE_MAX_AVAIL, 0, 0, 1024, 768 );       

  err = dc1394_feature_set_mode(camera_, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_AUTO);
  //err = dc1394_feature_set_mode(camera_, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_MANUAL);
  //err = dc1394_feature_set_value(camera_, DC1394_FEATURE_SHUTTER, 1000);
  //err = dc1394_feature_set_value(camera_, DC1394_FEATURE_SHUTTER, 500);
  //err = dc1394_feature_set_value(camera_, DC1394_FEATURE_GAIN, 250); // 210
  DC1394_ERR_CLN(err, CleanupAndExit(camera_), "Could not set exposure\n");

  // enable external trigger
  if (use_external_trigger_) {
    // not needed
    ////dc1394_external_trigger_set_source(camera_, DC1394_TRIGGER_SOURCE_0);
    //dc1394_external_trigger_set_polarity(camera_, DC1394_TRIGGER_ACTIVE_HIGH);
    dc1394_external_trigger_set_mode(camera_, DC1394_TRIGGER_MODE_0);
    dc1394_external_trigger_set_power(camera_, DC1394_ON);
  }

  //#define NUM_BUFFERS 8
  err = dc1394_capture_setup(camera_, ring_buffer_size_, DC1394_CAPTURE_FLAGS_DEFAULT);
  DC1394_ERR_CLN(err, CleanupAndExit(camera_), "Could not setup camera - "
                 "make sure that the video mode and framerate are\nsupported by your camera\n");

  // report camera's features
  dc1394featureset_t features;
  err=dc1394_feature_get_all(camera_, &features);
  if (err != DC1394_SUCCESS)
    dc1394_log_warning("Could not get feature set");
  else
    dc1394_feature_print_all(&features, stdout);

  // have the camera start sending us data
  err = dc1394_video_set_transmission(camera_, DC1394_ON);
  DC1394_ERR_CLN(err, CleanupAndExit(camera_), "Could not start camera iso transmission\n");

  uint32_t pix_depth;
  dc1394_format7_get_data_depth(camera_, video_mode_, &pix_depth);
  std::cout << "Pixel depth = " << pix_depth << "\n";

  dc1394_get_image_size_from_video_mode(camera_, video_mode_, &width_, &height_);
  std::cout << "WxH = " << width_ << "x" << height_ << "\n";

  dc1394color_filter_t cfilter;
  dc1394_format7_get_color_filter(camera_, video_mode_, &cfilter);
  printf("Bayer type = ");
  LibdcHelper::PrintColorFilter(cfilter);
  printf("\n");
}

CameraCaptureLibdc1394::~CameraCaptureLibdc1394() {
  dc1394_video_set_transmission(camera_, DC1394_OFF);
  dc1394_capture_stop(camera_);
  dc1394_camera_free(camera_);
}


uint64_t CameraCaptureLibdc1394::Grab(cv::Mat* left_img, cv::Mat* right_img) {
  dc1394video_frame_t *frame = nullptr;
  // get all filled buffers
  //uint32_t min_diff = std::numeric_limits<int>::max();
  int nearest_frame = -1;
  bool buffer_full = true;
  for (int f = 0; f < (int)ring_buffer_size_; f++) {
    dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_POLL, &frames_[f]);
    if (frames_[f] == nullptr) {
      buffer_full = false;
      //printf("[%d]: NULL\n", f);
    }
    else {
      nearest_frame = f;
    }
    //else {
    //  printf("[%d]: time = %lu\n", f, frames_[f]->timestamp);
    //  uint32_t diff = std::abs(static_cast<int>(frames_[f]->timestamp - time));
    //  if (diff < min_diff) {
    //    min_diff = diff;
    //    nearest_frame = f;
    //  }
    //}
  }
  // assure that we never drop new frames
  // if this happens we need a separate thread to handle the camera
  if (buffer_full) {
    printf("Buffer full!\n");
    throw 1;
  }
  bool empty_buffer = false;
  // if buffer was empty wait for a new frame
  if (nearest_frame == -1) {
    empty_buffer = true;
    dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_WAIT, &frame);
  }
  // otherwise take the nearest frame
  else {
    frame = frames_[nearest_frame];
  }
  //dc1394video_frame_t *frame = nullptr;
  // take control of the frame buffer
  //dc1394error_t err = dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_WAIT, &frame);
  //DC1394_ERR_CLN_RTN(err, CleanupAndExit(camera_), "Could not capture a frame\n");

  //printf("cc = %d\n", frame->color_coding);
  //printf("tb = %lu\n", frame->total_bytes);
  //printf("%d x %d\n", frame->size[0], frame->size[1]);
  //printf("Bytes per pixel = %d\n", frame->image_bytes / (width_  * height_));

  //printf("Frames behind = %d\n", rgb_frame->frames_behind);

  if (frame->color_coding == DC1394_COLOR_CODING_MONO16) {
    left_img->create(height_, width_, CV_8U);
    right_img->create(height_, width_, CV_8U);
    for (uint32_t i = 0; i < height_; i++) {
      for (uint32_t j = 0; j < width_; j++) {
        uint32_t idx = (i * width_ + j) * 2;
        left_img->at<uint8_t>(i,j) = ((uint8_t*)(frame->image))[idx];
        right_img->at<uint8_t>(i,j) = ((uint8_t*)(frame->image))[idx + 1];
      }
    }
  }
  else throw 1;

  uint64_t timestamp = frame->timestamp;
  // release the frame buffer
  // if buffer was empty only one frame to release
  if (empty_buffer) {
    dc1394_capture_enqueue(camera_, frame);
  }
  // else release all older filled frames in buffer
  else {
    for (int i = 0; i <= nearest_frame; i++) {
      if (frames_[i] != nullptr)
        dc1394_capture_enqueue(camera_, frames_[i]);
    }
  }
  return timestamp;
  ////frame->
  //// release the frame buffer
  //dc1394_capture_enqueue(camera_, frame);
  //return timestamp;
}

uint64_t CameraCaptureLibdc1394::Grab(cv::Mat& image) {
  dc1394video_frame_t *frame = nullptr;
  // take control of the frame buffer
  dc1394error_t err = dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_WAIT, &frame);

  //printf("cc = %d\n", frame->color_coding);
  //printf("tb = %lu\n", frame->total_bytes);
  //printf("%d x %d\n", frame->size[0], frame->size[1]);
  //printf("Bytes per pixel = %d\n", frame->image_bytes / (width_  * height_));

  DC1394_ERR_CLN_RTN(err, CleanupAndExit(camera_), "Could not capture a frame\n");
  //printf("Frames behind = %d\n", rgb_frame->frames_behind);

  if (frame->color_coding == DC1394_COLOR_CODING_MONO8) {
    image.create(height_, width_, CV_8U);
    for (uint32_t i = 0; i < height_; i++) {
      for (uint32_t j = 0; j < width_; j++) {
        image.at<uint8_t>(i,j) = frame->image[i*width_ + j];
      }
    }
  }
  else if (frame->color_coding == DC1394_COLOR_CODING_YUV422) {
    image.create(height_, width_, CV_8UC2);
    for (uint32_t i = 0; i < height_; i++) {
      for (uint32_t j = 0; j < width_; j++) {
        uint32_t idx = i * (2*width_) + (j*2);
        image.at<cv::Vec2b>(i,j)[0] = frame->image[idx];
        image.at<cv::Vec2b>(i,j)[1] = frame->image[idx+1];
      }
    }
  }
  else if (frame->color_coding == DC1394_COLOR_CODING_RAW16) {
    image.create(height_, width_, CV_16U);
    for (uint32_t i = 0; i < height_; i++) {
      for (uint32_t j = 0; j < width_; j++) {
        uint32_t idx = i * width_ + j;
        image.at<uint16_t>(i,j) = ((uint16_t*)(frame->image))[idx];
      }
    }
  }
  //frame->
  // release the frame buffer
  uint64_t timestamp = frame->timestamp;
  dc1394_capture_enqueue(camera_, frame);
  return timestamp;
}

//uint64_t CameraCaptureLibdc1394::Grab(cv::Mat& image) {
//  dc1394video_frame_t *frame = nullptr;
//  // take control of the frame buffer
//  dc1394error_t err = dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_WAIT, &frame);
//
//  //printf("cc = %d\n", frame->color_coding);
//  //printf("tb = %lu\n", frame->total_bytes);
//  //printf("tb = %d\n", frame->image_bytes);
//  //printf("%d x %d\n", frame->size[0], frame->size[1]);
//
//  DC1394_ERR_CLN_RTN(err, CleanupAndExit(camera_), "Could not capture a frame\n");
//  dc1394video_frame_t *rgb_frame = (dc1394video_frame_t*)calloc(1, sizeof(dc1394video_frame_t));
//  rgb_frame->color_coding = DC1394_COLOR_CODING_RGB8;
//  dc1394_convert_frames(frame, rgb_frame);
//  printf("Frames behind = %d\n", rgb_frame->frames_behind);
//
//  image.create(height_, width_, CV_8UC3);
//  //#pragma omp parallel for
//  for (uint32_t i = 0; i < height_; i++) {
//    for (uint32_t j = 0; j < width_; j++) {
//      uint32_t idx = i * (3*width_) + (j*3);
//      image.at<cv::Vec3b>(i,j)[2] = rgb_frame->image[idx];
//      image.at<cv::Vec3b>(i,j)[1] = rgb_frame->image[idx+1];
//      image.at<cv::Vec3b>(i,j)[0] = rgb_frame->image[idx+2];
//    }
//  }
//  free(rgb_frame->image);
//  free(rgb_frame);
//  // release the frame buffer
//  uint64_t timestamp = frame->timestamp;
//  dc1394_capture_enqueue(camera_, frame);
//  return timestamp;
//}

//uint64_t CameraCaptureLibdc1394::Grab(cv::Mat& image) {
//  dc1394video_frame_t *frame = nullptr;
//  // get all filled buffers
//  uint64_t max_timestamp = 0;
//  int newest_frame = -1;
//  bool buffer_full = true;
//  for (int f = 0; f < (int)ring_buffer_size_; f++) {
//    This is wrong
//    dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_POLL, &frames_[f]);
//    if (frames_[f] == nullptr) {
//      buffer_full = false;
//      printf("[%d]: NULL\n", f);
//      break;
//    }
//    else {
//      printf("[%d]: time = %lu\n", f, frames_[f]->timestamp);
//      if (frames_[f]->timestamp > max_timestamp) {
//        max_timestamp = frames_[f]->timestamp;
//        newest_frame = f;
//      }
//    }
//  }
//  // assure that we never drop new frames
//  // if this happens we need a separate thread to handle the camera
//  if (buffer_full) {
//    printf("Buffer full!\n");
//    throw 1;
//  }
//  bool empty_buffer = false;
//  // if buffer was empty wait for a new frame
//  if (newest_frame == -1) {
//    empty_buffer = true;
//    dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_WAIT, &frame);
//  }
//  // otherwise take the newest frame
//  else {
//    frame = frames_[newest_frame];
//  }
//
//  // covert YUV to RGB
//  dc1394video_frame_t *rgb_frame = (dc1394video_frame_t*)calloc(1, sizeof(dc1394video_frame_t));
//  rgb_frame->color_coding = DC1394_COLOR_CODING_RGB8;
//  dc1394_convert_frames(frame, rgb_frame);
//  image.create(height_, width_, CV_8UC3);
//  // no efect with omp
//  //#pragma omp parallel for
//  printf("Total bytes = %lu\n", rgb_frame->total_bytes);
//  printf("Image bytes = %d\n", rgb_frame->image_bytes);
//  printf("Data in padding = %d\n", rgb_frame->data_in_padding);
//  printf("Padding bytes = %d\n", rgb_frame->padding_bytes);
//  for (uint32_t i = 0; i < height_; i++) {
//    for (uint32_t j = 0; j < width_; j++) {
//      uint32_t idx = i * (3*width_) + (j*3);
//      // save to cv::Mat (RGB to BGR ordering)
//      image.at<cv::Vec3b>(i,j)[2] = rgb_frame->image[idx];
//      image.at<cv::Vec3b>(i,j)[1] = rgb_frame->image[idx+1];
//      image.at<cv::Vec3b>(i,j)[0] = rgb_frame->image[idx+2];
//    }
//  }
//  printf("Allocated image bytes = %lu\n", rgb_frame->allocated_image_bytes);
//
//  //static int frame_num = 0;
//  //std::ofstream ofile("padding_" + std::to_string(frame_num) + ".txt");
//  //for (uint32_t i = rgb_frame->image_bytes; i < rgb_frame->allocated_image_bytes; i++) {
//  //  //printf("%d\n", rgb_frame->image[i]);
//  //  ofile << rgb_frame->image[i];
//  //}
//  //frame_num++;
//
//  free(rgb_frame->image);
//  free(rgb_frame);
//  uint64_t timestamp = frame->timestamp;
//
//  // release the frame buffer
//  // if buffer was empty only one frame to release
//  if (empty_buffer) {
//    dc1394_capture_enqueue(camera_, frame);
//  }
//  // else release all filled frames in buffer
//  else {
//    for (uint32_t i = 0; i < ring_buffer_size_; i++) {
//      if (frames_[i] != nullptr)
//        dc1394_capture_enqueue(camera_, frames_[i]);
//    }
//  }
//  return timestamp;
//}

uint64_t CameraCaptureLibdc1394::Grab(cv::Mat& image, uint64_t time) {
  dc1394video_frame_t *frame = nullptr;
  // get all filled buffers
  uint32_t min_diff = std::numeric_limits<int>::max();
  int nearest_frame = -1;
  bool buffer_full = true;
  for (int f = 0; f < (int)ring_buffer_size_; f++) {
    dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_POLL, &frames_[f]);
    if (frames_[f] == nullptr) {
      buffer_full = false;
      printf("[%d]: NULL\n", f);
    }
    else {
      printf("[%d]: time = %lu\n", f, frames_[f]->timestamp);
      uint32_t diff = std::abs(static_cast<int>(frames_[f]->timestamp - time));
      if (diff < min_diff) {
        min_diff = diff;
        nearest_frame = f;
      }
    }
  }
  // assure that we never drop new frames
  // if this happens we need a separate thread to handle the camera
  if (buffer_full) {
    printf("Buffer full!\n");
    throw 1;
  }
  bool empty_buffer = false;
  // if buffer was empty wait for a new frame
  if (nearest_frame == -1) {
    empty_buffer = true;
    dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_WAIT, &frame);
  }
  // otherwise take the nearest frame
  else {
    frame = frames_[nearest_frame];
  }
  // covert YUV to RGB
  dc1394video_frame_t *rgb_frame = (dc1394video_frame_t*)calloc(1, sizeof(dc1394video_frame_t));
  rgb_frame->color_coding = DC1394_COLOR_CODING_RGB8;
  dc1394_convert_frames(frame, rgb_frame);
  image.create(height_, width_, CV_8UC3);
  // no efect with omp
  //#pragma omp parallel for
  for (uint32_t i = 0; i < height_; i++) {
    for (uint32_t j = 0; j < width_; j++) {
      uint32_t idx = i * (3*width_) + (j*3);
      // save to cv::Mat (RGB to BGR ordering)
      image.at<cv::Vec3b>(i,j)[2] = rgb_frame->image[idx];
      image.at<cv::Vec3b>(i,j)[1] = rgb_frame->image[idx+1];
      image.at<cv::Vec3b>(i,j)[0] = rgb_frame->image[idx+2];
    }
  }
  free(rgb_frame->image);
  free(rgb_frame);
  uint64_t timestamp = frame->timestamp;

  // release the frame buffer
  // if buffer was empty only one frame to release
  if (empty_buffer) {
    dc1394_capture_enqueue(camera_, frame);
  }
  // else release all older filled frames in buffer
  else {
    for (int i = 0; i <= nearest_frame; i++) {
      if (frames_[i] != nullptr)
        dc1394_capture_enqueue(camera_, frames_[i]);
    }
  }
  return timestamp;
}

// for moonocular camera
//uint64_t CameraCaptureLibdc1394::Grab(cv::Mat& image) {
//  dc1394video_frame_t *frame = nullptr;
//  // take control of the frame buffer
//  dc1394error_t err = dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_WAIT, &frame);
//  DC1394_ERR_CLN_RTN(err, CleanupAndExit(camera_), "Could not capture a frame\n");
//
//  image.create(height_, width_, CV_8U);
//  #pragma omp parallel for
//  for (uint32_t i = 0; i < height_; i++) {
//    for (uint32_t j = 0; j < width_; j++) {
//      image.at<uint8_t>(i,j) = frame->image[i*width_ + j];
//      //image.at<uint8_t>(i,j) = ((uint16_t*)frame->image)[i*width_ + j];
//    }
//  }
//  // release the frame buffer
//  dc1394_capture_enqueue(camera_, frame);
//
//  return frame->timestamp;
//}

//uint64_t CameraCaptureLibdc1394::Grab(cv::Mat& image) {
//  dc1394video_frame_t *frame = nullptr;
//  uint32_t f;
//  // get all filled buffers
//  for (f = 0; f < ring_buffer_size_; f++) {
//    dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_POLL, &frames_[f]);
//
//    // debug
//    //printf("time = %d\n", frames_[f]);
//    //if (frames_[f] != nullptr)
//    //  dc1394_capture_enqueue(camera_, frames_[f]);
//
//    if (frames_[f] == nullptr)
//      break;
//    else
//      printf("time = %lu\n", frames_[f]->timestamp);
//  }
//  if (f > 0) {
//    // release all older buffers
//    if (f > 1) {
//      // we never get 2 buffers
//      throw 1;
//      for (uint32_t j = 0; j < (f-1); j++)
//        dc1394_capture_enqueue(camera_, frames_[j]);
//    }
//    // return the newest buffer
//    frame = frames_[f-1];
//  }
//  // wait if there was no filled buffers 
//  else {
//    dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_WAIT, &frame);
//  }
//  // debug
//  //dc1394_capture_dequeue(camera_, DC1394_CAPTURE_POLICY_WAIT, &frame);
//
//  image.create(height_, width_, CV_8U);
//  #pragma omp parallel for
//  for (uint32_t i = 0; i < height_; i++) {
//    for (uint32_t j = 0; j < width_; j++) {
//      image.at<uint8_t>(i,j) = frame->image[i*width_ + j];
//      //image.at<uint8_t>(i,j) = ((uint16_t*)frame->image)[i*width_ + j];
//    }
//  }
//  // release the frame buffer
//  dc1394_capture_enqueue(camera_, frame);
//
//  return frame->timestamp;
//}


}
