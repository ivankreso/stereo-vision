
#include "../../sba/sba_base.h"
#include "../../sba/sba_ceres.h"

int main()
{
  optim::BALProblemStereo sba_problem("/home/kivan/Projects/cv-stereo/scripts/stereo_model_sba/SBA_dataset.txt", false);
  optim::SBAceres sba(&sba_problem);

  sba.runSBA();

  const double* params = sba_problem.mutable_cameras();
  for(int i = 0; i < sba_problem.num_cameras()*sba_problem.camera_block_size(); i++) {
    std::cout << params[i] << "\n";
  }
  core::Point pt1(10.0, 20.0);
  core::Point pt2;
  pt2 = pt1;
  std::cout << pt2 << "\n";

  return 0;
}
