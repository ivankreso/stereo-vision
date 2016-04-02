#ifndef CPP_GEMS_H_
#define CPP_GEMS_H_

#include <iostream>
#include <vector>
#include <array>

#include "guylib.h"

template <typename T>
std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
template <typename T>
std::vector<T>& operator+=(std::vector<T>& vector, const T& object) {
  vector.push_back(object);
  return vector;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
    os << " " << *ii;
  os << "]";
  return os;
}

namespace type {

template <typename T>
class Array2d {
 public:
  Array2d() {}
  Array2d(size_t rows, size_t cols) {
    resize(rows, cols);
  }
  void clear() {
    data_.clear();
  }
  void resize(size_t rows, size_t cols) {
    data_.resize(rows);
    for (size_t i = 0; i < rows; i++)
      data_[i].resize(cols);
  }
  std::vector<T>& operator[](size_t i) {
    return data_[i];
  }
 private:
  std::vector<std::vector<T>> data_;
};

}

#endif  // CPP_GEMS_H_
