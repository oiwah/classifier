#include <confidence_weighted/cw.h>

#include <cmath>
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
  Update(datum.category, scores, datum.fv);
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

double CW::CalcV(const feature_vector& fv,
                 const std::string& correct,
                 const std::string& non_correct_predict) {
  double v = 0.0;
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    if (cov_[correct].find(it->first) == cov_[correct].end())
      cov_[correct][it->first] = 1.0;
    v += cov_[correct][it->first] * it->second * it->second;

    if (non_correct_predict == non_class) continue;

    if (cov_[non_correct_predict].find(it->first) == cov_[non_correct_predict].end())
      cov_[non_correct_predict][it->first] = 1.0;
    v += cov_[non_correct_predict][it->first] * it->second * it->second;
  }

  return v;
}

double CW::CalcAlpha(double m, double v) const {
  double gamma = 0.0;
  double tmp = 1 + 2 * phi_ * m;
  gamma = - tmp + std::sqrt( tmp * tmp - (8 * phi_ * (m - phi_ * v)) );
  gamma /= (4 * phi_ * v);

  return std::max(0.0, gamma);
}

void CW::Update(const std::string& correct,
                const score2class& scores,
                const feature_vector& fv) {
  std::string non_correct_predict;
  double m = - CalcLossScore(scores, correct, &non_correct_predict);
  double v = CalcV(fv, correct, non_correct_predict);
  double alpha = CalcAlpha(m, v);

  if (alpha > 0.0) {
    for (feature_vector::const_iterator it = fv.begin();
         it != fv.end();
         ++it) {
      weight_[correct][it->first] += alpha * cov_[correct][it->first] * it->second;
      double tmp = 1 / cov_[correct][it->first] + 2 * alpha * phi_ * it->second * it->second;
      cov_[correct][it->first] = 1 / tmp;

      if (non_correct_predict == non_class) continue;
      
      weight_[non_correct_predict][it->first] -= alpha * cov_[non_correct_predict][it->first] * it->second;
      tmp = 1 / cov_[non_correct_predict][it->first] + 2 * alpha * phi_ * it->second * it->second;
      cov_[non_correct_predict][it->first] = 1 / tmp;
    }
  }
}

void CW::GetFeatureWeight(const std::string& feature,
                          std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature, weight_, results);
}

} //namespace
} //namespace
