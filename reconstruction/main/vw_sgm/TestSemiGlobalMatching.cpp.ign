#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/FileIO.h>
#include <gtest/gtest.h>

#include <SemiGlobalMatching.h>

using namespace vw;

TEST( SemiGlobalMatching, Cones ) {
  ImageView<PixelGray<uint8> > left_image, right_image;
  read_image(left_image,"data/cones/im2.png");
  read_image(right_image,"data/cones/im6.png");
  DiskImageView<uint8> disparity_truth("data/cones/disp2.png");

  write_image( "my_cone_disp.png",
               semi_global_matching_func( left_image, right_image ) );
}

TEST( SemiGlobalMatching, Teddy ) {
  ImageView<PixelGray<uint8> > left_image, right_image;
  read_image(left_image,"data/teddy/im2.png");
  read_image(right_image,"data/teddy/im6.png");
  DiskImageView<uint8> disparity_truth("data/teddy/disp2.png");

  write_image( "my_teddy_disp.png",
               semi_global_matching_func( left_image, right_image ) );
}

TEST( SemiGlobalMatching, Tsukuba ) {
  ImageView<PixelGray<uint8> > left_image, right_image;
  read_image(left_image,"data/tsukuba/scene1.row3.col3.png");
  read_image(right_image,"data/tsukuba/scene1.row3.col4.png");

  write_image( "my_tsukuba_disp.png",
               semi_global_matching_func( left_image, right_image ) );
}

TEST( SemiGlobalMatching, MOC ) {
  DiskImageView<PixelGray<uint8> > left_image("data/moc/epi-L.tif");
  DiskImageView<PixelGray<uint8> > right_image("data/moc/epi-R.tif");

  ImageViewRef<uint8> result =
    semi_global_matching( crop(edge_extend(left_image),-20,0,left_image.cols()+20,left_image.rows()),
                          edge_extend(right_image,0,0,left_image.cols()+20,left_image.rows()) );

  // Valid disparity is : Vector2(-9,-3)-Vector2(10,3)
  boost::scoped_ptr<DiskImageResource> r(DiskImageResource::create("my_moc_disp.tif",result.format()));
  r->set_block_write_size( Vector2i(256,256) );
  block_write_image( *r, result,
                     TerminalProgressCallback( "", "Rendering: ") );
}

int main( int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
