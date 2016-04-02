#include "video_grabber_bb.h"

double VideoGrabberBB::getStereoPair(CameraImage& imgLeft, CameraImage& imgRight)
{
   Image raw_image;
   // Retrieve an image
   error = cam.RetrieveBuffer(&raw_image);
   if (error != PGRERROR_OK)
   {
      PrintError(error);
      continue;
   }
}

