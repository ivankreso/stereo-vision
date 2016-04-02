#ifndef CORE_EIGEN_H_
#define CORE_EIGEN_H_

#include "Eigen/Core"

namespace core {

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<double,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor> Matrix;
typedef Eigen::Map<Vector> VectorRef;
typedef Eigen::Map<Matrix> MatrixRef;
typedef Eigen::Map<const Vector> ConstVectorRef;
typedef Eigen::Map<const Matrix> ConstMatrixRef;

// Column major matrices for DenseSparseMatrix/DenseQRSolver
typedef Eigen::Matrix<double,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::ColMajor> ColMajorMatrix;

typedef Eigen::Map<ColMajorMatrix, 0,
                   Eigen::Stride<Eigen::Dynamic, 1> > ColMajorMatrixRef;

typedef Eigen::Map<const ColMajorMatrix,
                   0,
                   Eigen::Stride<Eigen::Dynamic, 1> > ConstColMajorMatrixRef;



// C++ does not support templated typdefs, thus the need for this
// struct so that we can support statically sized Matrix and Maps.
template <int num_rows = Eigen::Dynamic, int num_cols = Eigen::Dynamic>
struct EigenTypes {
  typedef Eigen::Matrix <double, num_rows, num_cols, Eigen::RowMajor>
  Matrix;

  typedef Eigen::Map<
    Eigen::Matrix<double, num_rows, num_cols, Eigen::RowMajor> >
  MatrixRef;

  typedef Eigen::Matrix <double, num_rows, 1>
  Vector;

  typedef Eigen::Map <
    Eigen::Matrix<double, num_rows, 1> >
  VectorRef;


  typedef Eigen::Map<
    const Eigen::Matrix<double, num_rows, num_cols, Eigen::RowMajor> >
  ConstMatrixRef;

  typedef Eigen::Map <
    const Eigen::Matrix<double, num_rows, 1> >
  ConstVectorRef;
};

}  // namespace core

#endif  // CORE_EIGEN_H_
