#include "video_grabber_bb.h"

#include <cassert>

using namespace std;
using namespace FlyCapture2;

VideoGrabberBB::VideoGrabberBB(int board, int channel)
{
   //Camera cam;
   PGRGuid guid;
   //error_ error_;
   BusManager busMgr;
   FC2Config fcon;
   Format7ImageSettings fmt7ImageSettings;
   Format7PacketInfo fmt7PacketInfo;
   Format7Info fmt7Info;
   bool supported;
   bool valid;

   busMgr.GetCameraFromIndex(channel, &guid);
   error_ = cam_.Connect(&guid);
   if (error_ != PGRERROR_OK)
      error_.PrintErrorTrace();

   fmt7Info.mode = MODE_3;
   error_ = cam_.GetFormat7Info(&fmt7Info, &supported);
   if (error_ != PGRERROR_OK)
      error_.PrintErrorTrace();

   fmt7ImageSettings.mode = MODE_3;
   fmt7ImageSettings.offsetX = 0;
   fmt7ImageSettings.offsetY = 0;
   fmt7ImageSettings.width = fmt7Info.maxWidth;
   fmt7ImageSettings.height = fmt7Info.maxHeight;
   fmt7ImageSettings.pixelFormat = PIXEL_FORMAT_MONO16;

   error_ = cam_.ValidateFormat7Settings(&fmt7ImageSettings, &valid, &fmt7PacketInfo);
   error_ = cam_.SetFormat7Configuration(&fmt7ImageSettings, fmt7PacketInfo.recommendedBytesPerPacket);

   TriggerMode triggerMode;
   triggerMode.onOff = false;
   triggerMode.mode = 14;
   triggerMode.parameter = 0;
   triggerMode.source = 7;
   cam_.SetTriggerMode(&triggerMode);

   error_ = cam_.StartCapture();
   if (error_ != PGRERROR_OK)
      error_.PrintErrorTrace();

   CameraInfo cam_info;
   cam_.GetCameraInfo(&cam_info);
   printCameraInfo(&cam_info);
   //cout << cam_info.sensorResolution << "\n";
}


VideoGrabberBB::~VideoGrabberBB()
{
   error_ = cam_.StopCapture();
   if(error_ != PGRERROR_OK) {
      error_.PrintErrorTrace();
      return;
   }
   // Disconnect the camera
   //error_ = cam.Disconnect();
   //if(error_ != PGRERROR_OK)
   //   error_.PrintErrorTrace();
}


double VideoGrabberBB::getStereoImage(CameraImage& imgLeft, CameraImage& imgRight)
{
   Image img;
   // Retrieve an image
   error_ = cam_.RetrieveBuffer(&img);
   if (error_ != PGRERROR_OK)
      error_.PrintErrorTrace();
   
   // save timestamps...
   //TimeStamp stamp = img.GetTimeStamp();
   //cout << "sec: " << stamp.seconds << "\nmsec: " << stamp.microSeconds << "\n";

   // test save img
   //error_ = img.Save("slika.pgm");
   //if (error_ != PGRERROR_OK)
   //   error_.PrintErrorTrace();

   //cout << "(rows, cols): " << img.GetRows() << ", " << img.GetCols() << "\n";
   //cout << "stride: " << img.GetStride() << "\n";
   //cout << "data size: " << img.GetDataSize() << "\n";
   //for(int i = 0; i < 10; i++) {
   //   cout << static_cast<void*>(img(0,i)) << " --> ";
   //   cout << static_cast<int>(*img(0,i)) << "\n";
   //   cout << static_cast<void*>(img(0,i)+1) << " --> ";
   //   cout << static_cast<int>(*(img(0,i)+1)) << "\n";
   //}
   splitStereoImage(img, imgLeft, imgRight);
   return 42;
}

void VideoGrabberBB::splitStereoImage(Image& stereo_img, CameraImage& img_left, CameraImage& img_right)
{
   assert(stereo_img.GetCols() == img_left.cols_ && stereo_img.GetRows() == img_left.rows_);
   assert(stereo_img.GetCols() == img_right.cols_ && stereo_img.GetRows() == img_right.rows_);

   for(uint32_t i = 0; i < stereo_img.GetRows(); i++) {
      for(uint32_t j = 0; j < stereo_img.GetCols(); j++) {
         img_left(i,j) = (uint8_t)*stereo_img(i,j);
         img_right(i,j) = (uint8_t)*(stereo_img(i,j)+1);
      }
   }
}


void VideoGrabberBB::printCameraInfo(CameraInfo* cam_info)
{
    printf("\n*** CAMERA INFORMATION ***\n"
        "Serial number - %u\n"
        "Camera model - %s\n"
        "Camera vendor - %s\n"
        "Sensor - %s\n"
        "Resolution - %s\n"
        "Driver - %s\n"
        "Firmware version - %s\n"
        "Firmware build time - %s\n\n",
        cam_info->serialNumber,
        cam_info->modelName,
        cam_info->vendorName,
        cam_info->sensorInfo,
        cam_info->sensorResolution,
        cam_info->driverName,
        cam_info->firmwareVersion,
        cam_info->firmwareBuildTime);
    cout << "Max bus speed: " << cam_info->maximumBusSpeed << "\n";
    cout << "PCIE bus speed " << cam_info->pcieBusSpeed << "\n";
    //VideoMode video_mode;
    //FrameRate frame_rate;
    ///cam_.GetVideoModeAndFrameRate(&video_mode, &frame_rate);
    //cout << video_mode << "\n";
    //cout << frame_rate << "\n";
    // Retrieve frame rate property
    Property frmRate;
    frmRate.type = FRAME_RATE;
    error_ = cam_.GetProperty( &frmRate );
    if (error_ != PGRERROR_OK)
      error_.PrintErrorTrace();

    cout << "Frame rate: " << frmRate.absValue << " fps\n";
}