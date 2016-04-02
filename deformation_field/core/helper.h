#ifndef CORE_HELPER_H_
#define CORE_HELPER_H_

#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
//#include <opencv2/core/core.hpp>

namespace core {

class Helper
{
 public:
   static void CleanFolder(const std::string& folder);
};

inline
void Helper::CleanFolder(const std::string& folder)
{  
  //if(stat(folder.c_str(), &st) != -1) {
  //  std::cout << "Cleaning folder: " + folder << '\n';
  //  // rmdir removes only empty folders
  //  rmdir(folder.c_str());
  //}
  //if(stat(folder.c_str(), &st) == -1)
  //  mkdir(folder.c_str(), 0740);

  struct stat st = {0};
  if(stat(folder.c_str(), &st) != -1) {
    std::cout << "Cleaning folder: " + folder << '\n';
    std::string cmd = "rm -r " + folder;
    system(cmd.c_str());
  }
  mkdir(folder.c_str(), 0740);
}

} // end namespace

#endif
