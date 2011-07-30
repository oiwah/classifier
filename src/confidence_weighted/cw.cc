#include <confidence_weighted/cw.h>

#include <cmath>
#include <cfloat>
#include <algorithm>

namespace classifier {
namespace cw {
CW::CW(double phi) : phi_(phi) {
  weight_matrix().swap(weight_);
  covariance_matrix().swap(cov_);
}

void CW::Train(const datum& datum) {
  score2class scores(0);
  CalcScores(datum.fv, &scores);
  Update(datum, scores);
}

void CW::Train(const std::vector<datum>& data,
               const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (std::vector<datum>::const_iterator it = data.begin();
         it != data.end();
         ++it) {
      Train(*it);
    }
  }
}

void CW::Test(const feature_vector& fv,
              std::string* predict) const {
  score2class scores(0);
  CalcScores(fv, &scores);
  *predict = scores[0].second;
}

void CW::CalcScores(const feature_vector& fv,
                    score2class* scores) const {
  scores->push_back(make_pair(non_class_score, non_class));

  for (weight_matrix::const_iterator it = weight_.begin();
       it != weight_.end();
       ++it) {
    double score = InnerProduct(fv, it->second);
    scores->push_back(make_pair(score, it->first));
  }

  sort(scores->begin(), scores->end(),
       std::greater<std::pair<double, std::string> >());
}

double CW::CalcV(const datum& datum,
                 const std::string& non_correct_predict) {
  double v = 0.0;
  covariance_vector &correct_cov = cov_[datum.category];
  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    if (correct_cov.size() <= it->first)
      correct_cov.resize(it->first + 1, 1.0);
    v += correct_cov[it->first] * it->second * it->second;
  }

  if (non_correct_predict == non_class)
    return v;

  covariance_vector &wrong_cov = cov_[non_correct_predict];
  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    if (wrong_cov.size() <= it->first)
      wrong_cov.resize(it->first + 1, 1.0);
    v += wrong_cov[it->first] * it->second * it->second;
  }

  return v;
}

double CW::CalcAlpha(double m, double v) const {
  double gamma = 0.0;
  double tmp = 1.0 + 2.0 * phi_ * m;
  gamma = - tmp + std::sqrt( tmp * tmp - (8.0 * phi_ * (m - phi_ * v)) );
  gamma /= (4.0 * phi_ * v);

  return std::max(0.0, gamma);
}

void CW::Update(const datum& datum,
                const score2class& scores) {
  std::string non_correct_predict;
  double m = - CalcLossScore(scores, datum.category, &non_correct_predict);
  double v = CalcV(datum, non_correct_predict);
  double alpha = CalcAlpha(m, v);

  if (alpha > 0.0) {
    weight_vector &correct_weight = weight_[datum.category];
    covariance_vector &correct_cov = cov_[datum.category];
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      if (correct_weight.size() <= it->first)
        correct_weight.resize(it->first + 1, 0.0);
      correct_weight[it->first] += alpha * correct_cov[it->first] * it->second;

      double tmp = 1.0 / correct_cov[it->first]
          + 2.0 * alpha * phi_ * it->second * it->second;
      if (1.0 < tmp && tmp < 10.0E100)
          correct_cov[it->first] = 1.0 / tmp;
    }

    if (non_correct_predict == non_class)
      return;

    weight_vector &wrong_weight = weight_[non_correct_predict];
    covariance_vector &wrong_cov = cov_[non_correct_predict];
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      if (wrong_weight.size() <= it->first)
        wrong_weight.resize(it->first + 1, 0.0);
      wrong_weight[it->first] -= alpha * wrong_cov[it->first] * it->second;

      double tmp = 1.0 / wrong_cov[it->first]
          + 2.0 * alpha * phi_ * it->second * it->second;
      if (1.0 < tmp && tmp < 10.0E100)
          wrong_cov[it->first] = 1.0 / tmp;
    }
  }
}

void CW::GetFeatureWeight(size_t feature_id,
                          std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature_id, weight_, results);
}

} //namespace
} //namespace
