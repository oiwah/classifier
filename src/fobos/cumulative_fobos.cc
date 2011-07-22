#include <fobos/cumulative_fobos.h>

#include <cmath>
#include <algorithm>

namespace classifier {
namespace fobos {
CumulativeFOBOS::CumulativeFOBOS(double eta, double lambda) : dataN_(0), eta_(eta), lambda_(lambda), truncate_sum_(0.0) {
  weight_matrix().swap(weight_);
  weight_matrix().swap(prev_truncate_);
}

void CumulativeFOBOS::Train(const datum& datum, bool truncate) {
  ++dataN_;
  Truncate(datum.fv);

  Update(datum);
  if (truncate)
    TruncateAll();
}

void CumulativeFOBOS::Train(const std::vector<datum>& data,
                            const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      Train(data[i], false);
    }
  }
  TruncateAll();
}

void CumulativeFOBOS::Test(const feature_vector& fv,
                           std::string* predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  CalcScores(fv, &score2class);
  *predict = score2class[0].second;
}

void CumulativeFOBOS::Truncate(const feature_vector& fv) {
  for (feature_vector::const_iterator fv_it = fv.begin();
       fv_it != fv.end();
       ++fv_it) {
    for (weight_matrix::const_iterator wm_it = weight_.begin();
         wm_it != weight_.end();
         ++wm_it) {
      weight_vector prev_truncate_vector = prev_truncate_[wm_it->first];

      double prev_truncate_value = 0.0;
      if (prev_truncate_vector.find(fv_it->first) != prev_truncate_vector.end())
        prev_truncate_value = prev_truncate_vector[fv_it->first];

      double weight_value = weight_[wm_it->first][fv_it->first];
      double truncate_value = - prev_truncate_value;

      if (weight_value > 0.0) {
        truncate_value -= truncate_sum_;
        if (weight_value + truncate_value < 0.0)
          truncate_value = - weight_value;
      } else {
        truncate_value += truncate_sum_;
        if (weight_value + truncate_value > 0.0)
          truncate_value = - weight_value;
      }
      weight_[wm_it->first][fv_it->first] += truncate_value;
      prev_truncate_[wm_it->first][fv_it->first] += truncate_value;
    }
  }
  truncate_sum_ += lambda_ * eta_ / (std::sqrt(dataN_) * 2.0);
}

void CumulativeFOBOS::TruncateAll() {
  for (weight_matrix::const_iterator wm_it = weight_.begin();
       wm_it != weight_.end();
       ++wm_it) {
    weight_vector wv = wm_it->second;
    for (weight_vector::const_iterator wv_it = wv.begin();
         wv_it != wv.end();
         ++wv_it) {
      double truncate_value = prev_truncate_[wm_it->first][wv_it->first];
      if (wv_it->second > 0.0) {
        truncate_value -= truncate_sum_;
        if (wv_it->second + truncate_value < 0.0)
          truncate_value = - wv_it->second;
      } else {
        truncate_value += truncate_sum_;
        if (wv_it->second + truncate_value > 0.0)
          truncate_value = - wv_it->second;
      }
      weight_[wm_it->first][wv_it->first] += truncate_value;
      prev_truncate_[wv_it->first][wv_it->first] += truncate_value;
    }
  }
}

void CumulativeFOBOS::Update(const datum& datum) {
  std::string non_correct_predict;
  double hinge_loss = CalcHingeLoss(datum, &non_correct_predict);

  if (hinge_loss > 0.0) {
    double step_distance = eta_ / (std::sqrt(dataN_) * 2.0);

    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      weight_[datum.category][it->first] += step_distance * it->second;
      if (non_correct_predict != non_class)
        weight_[non_correct_predict][it->first] -= step_distance * it->second;
    }
  }
}

void CumulativeFOBOS::CalcScores(const feature_vector& fv,
                                 std::vector<std::pair<double, std::string> >* score2class) const {
  score2class->push_back(make_pair(non_class_score, non_class));

  for (weight_matrix::const_iterator it = weight_.begin();
       it != weight_.end();
       ++it) {
    weight_vector wv = it->second;
    double score = InnerProduct(fv, &wv);
    score2class->push_back(make_pair(score, it->first));
  }

  sort(score2class->begin(), score2class->end(),
       std::greater<std::pair<double, std::string> >());
}

double CumulativeFOBOS::CalcHingeLoss(const datum& datum,
                                      std::string* non_correct_predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  CalcScores(datum.fv, &score2class);

  bool correct_done = false;
  bool predict_done = false;
  double score = 1.0;
  for (std::vector<std::pair<double, std::string> >::const_iterator
         it = score2class.begin();
       it != score2class.end();
       ++it) {
    if (it->second == datum.category) {
      score -= it->first;
      correct_done = true;
    } else if (!predict_done) {
      *non_correct_predict = it->second;
      if (*non_correct_predict != non_class)
        score += it->first;
      predict_done = true;
    }

    if (correct_done && predict_done)
      break;
  }

  return score;
}

void CumulativeFOBOS::GetFeatureWeight(const std::string& feature,
                                       std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature, weight_, results);
}

} //namespace
} //namespace
