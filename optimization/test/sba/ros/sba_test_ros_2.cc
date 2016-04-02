#include "../../sba/extern/ros_sba/include/sba/sba.h"
#include "../../sba/extern/ros_sba/include/sba/sba_file_io.h"

#include <eigen3/Eigen/Core>

using namespace sba;
using namespace std;

void processSBAfile(char* filename)
{
    // Create an empty SBA system.
    SysSBA sys;
    
    // Read in information from the bundler file.
    readBundlerFile(filename, sys);
    
    // Provide some information about the data read in.
    cout << "Cameras (nodes): " << (int)sys.nodes.size() << 
            ", Points: " << (int)sys.tracks.size() << "\n\n";
    for(int i = 0; i < sys.nodes.size(); i++) {
      //Eigen::Matrix<double,4,1> trans;
      //std::cout << trans << endl;
      std::cout << sys.nodes[i].trans << "\n"; 
      std::cout << sys.nodes[i].qrot.toRotationMatrix() << "\n\n";
    }
    
    // Perform SBA with 10 iterations, an initial lambda step-size of 1e-3, 
    // and using CSPARSE.
    sys.doSBA(10, 1e-3, 1);

    for(int i = 0; i < sys.nodes.size(); i++) {
      //Eigen::Matrix<double,4,1> trans;
      //std::cout << trans << endl;
      std::cout << sys.nodes[i].trans << "\n"; 
      std::cout << sys.nodes[i].qrot.toRotationMatrix() << "\n\n";
    }

    int npts = sys.tracks.size();

//    ROS_INFO("Bad projs (> 10 pix): %d, Cost without: %f", 
//        (int)sys.countBad(10.0), sqrt(sys.calcCost(10.0)/npts));
//    ROS_INFO("Bad projs (> 5 pix): %d, Cost without: %f", 
//        (int)sys.countBad(5.0), sqrt(sys.calcCost(5.0)/npts));
//    ROS_INFO("Bad projs (> 2 pix): %d, Cost without: %f", 
//        (int)sys.countBad(2.0), sqrt(sys.calcCost(2.0)/npts));
    
    printf("Cameras (nodes): %d, Points: %d\n", (int)sys.nodes.size(), (int)sys.tracks.size());
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
      printf("Arguments are:  <input filename>");
      return -1;
    }
    char* filename = argv[1];
    
    processSBAfile(filename);
    
    return 0;
}
