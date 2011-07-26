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
    for (size_t i = 0; i < data.size(); ++i) {
      Train(data[i]);
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
    weight_vector wv = it->second;
    double score = InnerProduct(fv, &wv);
    scores->push_back(make_pair(score, it->first));
  }

  sort(scores->begin(), scores->end(),
       std::greater<std::pair<double, std::string> >());
}

double CW::CalcV(const datum& datum,
                 const std::string& non_correct_predict) {
  double v = 0.0;
  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    if (cov_[datum.category].find(it->first) == cov_[datum.category].end())
      cov_[datum.category][it->first] = 1.0;
    v += cov_[datum.category][it->first] * it->second * it->second;

    if (non_correct_predict == non_class) continue;

    if (cov_[non_correct_predict].find(it->first) == cov_[non_correct_predict].end())
      cov_[non_correct_predict][it->first] = 1.0;
    v += cov_[non_correct_predict][it->first] * it->second * it->second;
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
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      weight_[datum.category][it->first]
          += alpha * cov_[datum.category][it->first] * it->second;
      double tmp = 1.0 / cov_[datum.category][it->first]
          + 2.0 * alpha * phi_ * it->second * it->second;
      if (1.0 < tmp && tmp < 10.0E100)
          cov_[datum.category][it->first] = 1.0 / tmp;
      if (non_correct_predict == non_class) continue;

      weight_[non_correct_predict][it->first]
          -= alpha * cov_[non_correct_predict][it->first] * it->second;
      tmp = 1.0 / cov_[non_correct_predict][it->first]
          + 2.0 * alpha * phi_ * it->second * it->second;
      if (1.0 < tmp && tmp < 10.0E100)
          cov_[non_correct_predict][it->first] = 1.0 / tmp;
    }
  }
}

void CW::GetFeatureWeight(const std::string& feature,
                          std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature, weight_, results);
}

} //namespace
} //namespace
