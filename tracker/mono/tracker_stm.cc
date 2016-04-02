#include "tracker_stm.h"

#include <bitset>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../../core/math_helper.h"
#include "../../reconstruction/base/stereo_costs.h"

namespace track  {

namespace {

void draw_descriptors(const cv::Mat& descriptors) {
  int wsize = std::sqrt(descriptors.rows);
  int num_dec = descriptors.cols;
  cv::Size imgsz = cv::Size(200, 200);
  cv::Mat desc, desc_img;
  for(int i = 0; i < num_dec; i++) {
    desc = descriptors.col(i).clone().reshape(0, wsize);
    desc.convertTo(desc_img, CV_8U);
    cv::resize(desc_img, desc_img, imgsz, 0, 0, cv::INTER_NEAREST);
    cv::imshow("patch_" + std::to_string(i), desc_img);
  }
  cv::waitKey(0);
  cv::destroyAllWindows();
}

double get_mean_dist(const std::vector<double>& weights, const std::vector<double>& distances) {
  double mean_dist = 0.0;
  for(size_t i = 0; i < distances.size(); i++) {
    mean_dist += weights[i] * distances[i];
  }
  return mean_dist;
}

template<typename T>
void get_binary_values(T val, int elem_sz, std::vector<double>& bin_values) {
  std::bitset<8> bits(val);
  for(int i = 0; i < elem_sz; i++) {
    //std::cout << bits[i] << '\n';
    bin_values.push_back(static_cast<double>(bits[i]));
  }
}

cv::Mat get_dense_vector(const cv::Mat& mat) {
  assert(mat.cols > 1 && mat.rows == 1);
  int elem_sz = 8;
  cv::Mat ret(mat.cols * elem_sz, 1, CV_64F);
  for(int i = 0; i < mat.cols; i++) {
    uint8_t val = mat.at<uint8_t>(i);
    std::vector<double> bin_values;
    get_binary_values<uint8_t>(val, elem_sz, bin_values);
    for(int j = 0; j < elem_sz; j++) {
      ret.at<double>(i*elem_sz + j) = bin_values[j];
      //printf("%f\n\n", bin_values[j]);
    }
  }
  return ret;
}

void round_mat(cv::Mat& mat) {
  for(int i = 0; i < mat.rows; i++) {
    for(int j = 0; j < mat.cols; j++)
      mat.at<double>(i,j) = std::round(mat.at<double>(i,j));
  }
}

}

TrackerSTM::TrackerSTM(TrackerBase* tracker, double Q, double a) :
    tracker_(tracker), Q_(Q), a_(a) {
  int max_features = tracker->countFeatures();
  stm_.resize(max_features);
  stm_dists_.resize(max_features);
}

TrackerSTM::~TrackerSTM() {
  if(tracker_ != nullptr)
    delete tracker_;
}

int TrackerSTM::init(const cv::Mat& img) {
  tracker_->init(img);
  return 0;
}

int TrackerSTM::track(const cv::Mat& img) {
  tracker_->track(img);
  int alive_before = tracker_->countTracked();
  cv::Mat st_desc;
  cv::Mat double_stm;
  size_t num_outliers = 0;
  for(int i = 0; i < tracker_->countFeatures(); i++) {
    if(tracker_->isAlive(i)) {
      FeatureData fdata = tracker_->getFeatureData(i);
      // If the feature is new then we need to clear the STM and then add
      // first two descriptors to STM
      if(fdata.feat_.age_ == 1) {
        // for FREAK
        //cv::Mat vec_prev = get_dense_vector(fdata.desc_prev_);
        //cv::Mat vec_curr = get_dense_vector(fdata.desc_curr_);
        // for NCC
        //cv::Mat vec_prev = fdata.desc_prev_.clone();
        //cv::Mat vec_curr = fdata.desc_curr_.clone();
        //stm_[i] = vec_prev;

        stm_dists_[i].clear();
        //fdata.desc_prev_.convertTo(stm_[i], CV_64F);
        int num_rows = fdata.desc_curr_.rows * fdata.desc_curr_.rows;
        fdata.desc_prev_.clone().reshape(0, num_rows).convertTo(stm_[i], CV_64F);

        cv::Mat vec_curr;
        //fdata.desc_curr_.convertTo(vec_curr, CV_64F);
        fdata.desc_curr_.clone().reshape(0, num_rows).convertTo(vec_curr, CV_64F);

        //std::cout << fdata.desc_prev_ << "\n";
        //cv::hconcat(stm_[i], vec_prev, stm_[i]);
        cv::hconcat(stm_[i], vec_curr, stm_[i]);
        // TODO: optimization: have both dense and compressed representations of
        // STM matrix and then can use hamm
        //uint32_t dist = core::MathHelper::hammDist<uint32_t>(stm_[i].col(0), stm_[i].col(1));
        // get Hamming distance
        //double dist = core::MathHelper::GetDistanceL1<double>(stm_[i].col(0), stm_[i].col(1));
        double dist = core::MathHelper::GetDistanceNCC<double>(stm_[i].col(0), stm_[i].col(1));
        //double dist = core::MathHelper::
        //    GetDistanceChiSq<double>(stm_[i].col(0), stm_[i].col(1));
        dist = std::max(0.00001, dist);
        //printf("%f\n", dist);
        if(std::isnan(dist))
          throw "Error\n";
        stm_dists_[i].push_back(dist);
        stm_dists_[i].push_back(dist);

        // add NCC descriptors also
        //ncc_descriptors_.push_back(fdata.ncc_prev_);
        //ncc_descriptors_.push_back(fdata.ncc_curr_);
      }
      // else only add the descriptor in the current frame
      else {
        int len = stm_[i].cols;
        cv::Mat stm_coeffs_(len, 1, CV_64F);
        std::vector<double> weights;
        weights.resize(len);
        double weights_sum = 0.0;
        for(int j = 0; j < len; j++) {
          int t = j + 1;
          weights[j] = std::exp(static_cast<double>(-(t-len)*(t-len)) / (2.0*a_*a_));
          weights_sum += weights[j];
        }
        double L1_norm = 0.0;
        for(int j = 0; j < len; j++) {
          // normaize the weights
          weights[j] /= weights_sum;
          //printf("Weight %d = %f\n", j, weights[j]);
          //double coeff = weights[j] / stm_dists_[i][j];
          //if (distance_type == DistanceType::NCC)
            double coeff = weights[j] * stm_dists_[i][j];
          stm_coeffs_.at<double>(j) = coeff;
          L1_norm += coeff;
        }
        // normalize the coeffs
        for(int j = 0; j < len; j++) {
          stm_coeffs_.at<double>(j) /= L1_norm;
          //printf("STM dist %d = %f\n", j, stm_dists_[i][j]);
          //printf("STM coeff %d = %f\n", j, stm_coeffs_.at<double>(j));
        }

        //stm_[i].convertTo(double_stm, CV_64F);
        //st_desc = double_stm * stm_coeffs_;
        //std::cout << "\n\nSTM coeffs:\n" << stm_coeffs_;
        //std::cout << "STM:\n" << stm_[i];
        st_desc = stm_[i] * stm_coeffs_;
        //std::cout << stm_[i] << "\n\n" << stm_coeffs_ << "\n\n" << st_desc << "\n\n";
        // do the rounding
        //std::cout << "st_desc:\n" << st_desc;
        //round_mat(st_desc);
        //std::cout << "\n\n\nAfter round st_desc:\n" << st_desc;

        // for FREAK
        //cv::Mat curr_desc = get_dense_vector(fdata.desc_curr_);
        // for NCC
        //cv::Mat curr_desc;
        //fdata.desc_curr_.convertTo(curr_desc, CV_64F);
        int num_rows = fdata.desc_curr_.rows * fdata.desc_curr_.rows;
        cv::Mat curr_desc;
        fdata.desc_curr_.clone().reshape(0, num_rows).convertTo(curr_desc, CV_64F);

        //for(int i = 0; i < stm_[i].rows; i++) {
        //  printf("%f == %f\n", stm_[i].at<double>(i,0), st_desc.at<double>(i,0));
        //  // TODO check the computation of st_desc here
        //  //for(int j = 0; j < stm_[j]
        //}

        //double dist = core::MathHelper::GetDistanceL1<double>(st_desc, curr_desc);
        double dist = core::MathHelper::GetDistanceNCC<double>(st_desc, curr_desc);
        //double dist = core::MathHelper::GetDistanceChiSq<double>(st_desc, curr_desc);
        dist = std::max(0.00001, dist);
        if(std::isnan(dist))
          throw "Error\n";
        //uint32_t dist = core::MathHelper::hammDist<uint32_t>(st_desc_binary, fdata.desc_curr_);
        double w_mean_dist = get_mean_dist(weights, stm_dists_[i]);
        double delta = dist / w_mean_dist;

        //printf("dist / mean_dist = %f / %f\n", dist, w_mean_dist);
        //printf("Desc size = %d -- %f < %f\n", len, delta, Q_);
        //std::cout << delta << " < " << Q_ << "\n";
        if(delta < Q_) {
          cv::hconcat(stm_[i], curr_desc, stm_[i]);
          stm_dists_[i].push_back(dist);
          //ncc_descriptors_.push_back(fdata.ncc_curr_);
        }
        else {
          //tracker_->showTrack(i);
          //if(stm_[i].cols > 1) {
          //  draw_descriptors(stm_[i]);
          //}
          // remove this track
          tracker_->removeTrack(i);
          num_outliers++;
        }
        // TODO: clear descriptors in STM for which t < (n < 3a)
      }
    }
  }
  std::cout << "[TrackerSTM]: Number of outliers ratio = " << static_cast<double>(num_outliers) /
            alive_before * 100.0 << "%" << std::endl;
  return 0;
}

}
