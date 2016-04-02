#ifndef CALIB_HELPER_H_
#define CALIB_HELPER_H_

#include <iostream>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

namespace CalibHelper {

/* Returns a list of files in a directory (except the ones that begin with a dot) */
void GetFilesInFolder(const std::string& dir_path, std::vector<std::string>& files,
                      bool full_path = true) {
  DIR *dir;
  struct dirent *ent;
  struct stat st;
  dir = opendir(dir_path.c_str());
  while ((ent = readdir(dir)) != NULL) {
    const std::string file_name = ent->d_name;
    const std::string full_file_name = dir_path + "/" + file_name;
    if (file_name[0] == '.')
      continue;
    if (stat(full_file_name.c_str(), &st) == -1)
      continue;
    const bool is_directory = (st.st_mode & S_IFDIR) != 0;
    if (is_directory)
      continue;

    if (full_path)
      files.push_back(full_file_name);
    else
      files.push_back(file_name);
  }
  closedir(dir);
}

}   // namespace CalibHelper


#endif  // CALIB_HELPER_H_
