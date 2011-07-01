#include <confidence_weighted/cw.h>

#include <cfloat>
#include <cmath>
#include <algorithm>

namespace classifier {
namespace cw {
CW::CW(double phi) : phi_(phi) {
  weight_matrix().swap(weight_);
  covariance_matrix().swap(cov_);
}

void CW::Train(const std::vector<datum>& data,
               const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      std::vector<std::pair<double, std::string> > score2class(0);
      CalcScores(data[i].fv, &score2class);
      Update(data[i].category, score2class, data[i].fv);
    }
  }
}

void CW::Test(const feature_vector& fv,
              std::string* predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  CalcScores(fv, &score2class);
  *predict = score2class[0].second;
}

void CW::CalcScores(const feature_vector& fv,
                    std::vector<std::pair<double, std::string> >* score2class) const {
  score2class->push_back(make_pair(-DBL_MAX, non_class));

  for (weight_matrix::const_iterator it = weight_.begin();
       it != weight_.end();
       ++it) {
    weight_vector wv = it->second;
    double score = InnerProduct(fv, wv);
    score2class->push_back(make_pair(score, it->first));
  }

  sort(score2class->begin(), score2class->end(),
       std::greater<std::pair<double, std::string> >());
}

double CW::InnerProduct(const feature_vector& fv,
                        weight_vector& wv) const {
  double score = 0.0;
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it)
    score += wv[it->first] * it->second;

  return score;
}

double CW::CalcM(const std::vector<std::pair<double, std::string> >& score2class,
                 const std::string& correct,
                 std::string* non_correct_predict) const {
  bool correct_done = false;
  bool predict_done = false;
  double score = 0.0;
  for (std::vector<std::pair<double, std::string> >::const_iterator
           it = score2class.begin();
       it != score2class.end();
       ++it) {
    if (it->second == correct) {
      score += it->first;
      correct_done = true;
    } else if (!predict_done) {
      *non_correct_predict = it->second;
      if (*non_correct_predict != non_class)
        score -= it->first;
      predict_done = true;
    }

    if (correct_done && predict_done)
      break;
  }

  return score;
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
                const std::vector<std::pair<double, std::string> >& score2class,
                const feature_vector& fv) {
  std::string non_correct_predict;
  double m = CalcM(score2class, correct, &non_correct_predict);
  double v = CalcV(fv, correct, non_correct_predict);
  double alpha = CalcAlpha(m, v);

  if (alpha > 0.0) {
    for (feature_vector::const_iterator it = fv.begin();
         it != fv.end();
         ++it) {
      weight_[correct][it->first] += alpha * cov_[correct][it->first] * it->second;
      double tmp = 1 / cov_[correct][it->first] + 2 * alpha * phi_ * it->second * it->second;
      cov_[correct][it->first] = 1 / tmp;

      if (non_correct_predict != non_class) {
        weight_[non_correct_predict][it->first] -= alpha * cov_[non_correct_predict][it->first] * it->second;
        double tmp = 1 / cov_[non_correct_predict][it->first] + 2 * alpha * phi_ * it->second * it->second;
        cov_[non_correct_predict][it->first] = 1 / tmp;
      }
    }
  }
}

void CW::CompareFeatureWeight(const std::string& feature,
                              std::vector<std::pair<std::string, double> >* results) const {
  for (weight_matrix::const_iterator it = weight_.begin();
       it != weight_.end();
       ++it) {
    std::string category = it->first;
    if (it->second.find(feature) == it->second.end()) {
      results->push_back(make_pair(category, 0.0));
    } else {
      double score = it->second.at(feature);
      results->push_back(make_pair(category, score));
    }
  }
}

} //namespace
} //namespace
