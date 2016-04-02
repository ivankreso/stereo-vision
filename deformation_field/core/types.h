#ifndef CORE_TYPES_H_
#define CORE_TYPES_H_

#include <math.h>
#include <ostream>
#include <iomanip>
#include <cassert>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <opencv2/core/core.hpp>

namespace core {

struct Size {
  Size(int w, int h) : width_(w), height_(h) {}
  int width_;
  int height_;
};

struct Point {
  double x_;
  double y_;

  Point(double x=-1,double y=-1): x_(x), y_(y) {}
  Point(const Point& pt) { *this = pt; }

  double l1() const { return fabs(x_)+fabs(y_);}

  Point& operator=(const Point& pt) {
    x_ = pt.x_;
    y_ = pt.y_;
    return *this;
  }
  Point& operator+=(const Point& rhs){
    x_+=rhs.x_;
    y_+=rhs.y_;
    return *this;
  }
  Point& operator-=(const Point& rhs){
    x_-=rhs.x_;
    y_-=rhs.y_;
    return *this;
  }
 private:
  friend class boost::serialization::access;
  // When the class Archive corresponds to an output archive, the
  // & operator is defined similar to <<.  Likewise, when the class Archive
  // is a type of input archive the & operator is defined similar to >>.
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar & x_;
    ar & y_;
  }
};

//struct DescriptorNCC : public Descriptor
struct DescriptorNCC
{
  cv::Mat vec;
  double A, B, C;
  DescriptorNCC() : A(0.0), B(0.0), C(0.0) {}

  DescriptorNCC(const DescriptorNCC& other) {
    *this = other;
  }
  DescriptorNCC& operator=(const DescriptorNCC& other) {
    this->vec = other.vec.clone();
    this->A = other.A;
    this->B = other.B;
    this->C = other.C;
    return *this;
  }
  double compare(DescriptorNCC& desc) const {
    assert(vec.rows == desc.vec.rows);
    double n = vec.rows;
    double D = vec.dot(desc.vec);
    return (n * D - (A * desc.A)) * C * desc.C;
  }
  //virtual double compare(Descriptor& other) const {
  //  if(DescriptorNCC* desc = dynamic_cast<DescriptorNCC*>(&other)) {
  //    assert(vec.rows == desc->vec.rows);
  //    double n = vec.rows;
  //    double D = vec.dot(desc->vec);
  //    return (n * D - (A * desc->A)) * C * desc->C;
  //  }
  //  else
  //    throw "Cast error!\n";
  //}
};

inline Point operator+(Point lhs, const Point& rhs){
  lhs += rhs;
  return lhs;
}
inline Point operator-(Point lhs, const Point& rhs){
  lhs -= rhs;
  return lhs;
}
inline std::ostream& operator<<(std::ostream& out, const Point& pt)
{
  //out << std::setprecision(2) << "(" << pt.x_ << ", " << pt.y_ << ")";
  out << "(" << pt.x_ << ", " << pt.y_ << ")";
  return out;
}

}

#endif

