#include "fobos.h"

#include <cmath>
#include <algorithm>

namespace classifier {
namespace fobos {
FOBOS::FOBOS(double eta, double lambda) : dataN_(0), eta_(eta), lambda_(lambda), truncate_sum_(0.0) {
  weight_matrix().swap(weight_);
  weight_matrix().swap(prev_truncate_);
}

void FOBOS::Train(const datum& datum, bool truncate) {
  ++dataN_;
  Truncate(datum.fv);

  score2class scores(0);
  CalcScores(datum.fv, &scores);
  Update(datum, scores);

  if (truncate)
    TruncateAll();
}

void FOBOS::Train(const std::vector<datum>& data,
                  const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (std::vector<datum>::const_iterator it = data.begin();
         it != data.end();
         ++it) {
      Train(*it, false);
    }
  }
  TruncateAll();
}

void FOBOS::Test(const feature_vector& fv,
                 std::string* predict) const {
  score2class scores(0);
  CalcScores(fv, &scores);
  *predict = scores[0].second;
}

void FOBOS::Truncate(const feature_vector& fv) {
  for (weight_matrix::const_iterator wm_it = weight_.begin();
       wm_it != weight_.end();
       ++wm_it) {
    weight_vector &weight_vec = weight_[wm_it->first];
    weight_vector &truncate_vec = prev_truncate_[wm_it->first];
    for (feature_vector::const_iterator fv_it = fv.begin();
         fv_it != fv.end();
         ++fv_it) {
      if (fv_it->first < truncate_vec.size()) {
        if (truncate_vec[fv_it->first] == 0.0) continue;
        double truncate_value = truncate_sum_ - truncate_vec[fv_it->first];
        double weight_value = weight_vec[fv_it->first];

        if (weight_value > 0.0) {
          weight_vec[fv_it->first] = std::max(0.0,
                                              weight_value - truncate_value);
        } else {
          weight_vec[fv_it->first] = std::min(0.0,
                                              weight_value + truncate_value);
        }
      } else {
        truncate_vec.resize(fv_it->first + 1, 0.0);
        weight_vec.resize(fv_it->first + 1, 0.0);
      }
      truncate_vec[fv_it->first] = truncate_sum_;
    }
  }
  truncate_sum_ += lambda_ * eta_ / (std::sqrt(dataN_) * 2.0);
}

void FOBOS::TruncateAll() {
  for (weight_matrix::const_iterator wm_it = weight_.begin();
       wm_it != weight_.end();
       ++wm_it) {
    weight_vector &weight_vec = weight_[wm_it->first];
    weight_vector &truncate_vec = prev_truncate_[wm_it->first];

    for (size_t feature_id = 0; feature_id < weight_vec.size(); ++feature_id) {
      double prev_truncate = truncate_vec[feature_id];
      double truncate_value = truncate_sum_ - prev_truncate;
      double weight_value = weight_vec[feature_id];
      if (weight_value > 0.0) {
        weight_vec[feature_id] = std::max(0.0, weight_value - truncate_value);
      } else {
        weight_vec[feature_id] = std::min(0.0, weight_value + truncate_value);
      }
      truncate_vec[feature_id] = truncate_sum_;
    }
  }
}

void FOBOS::Update(const datum& datum,
                   const score2class& scores) {
  std::string non_correct_predict;
  double hinge_loss = CalcLossScore(scores, datum.category, &non_correct_predict, 1.0);

  if (hinge_loss > 0.0) {
    double step_distance = eta_ / (std::sqrt(dataN_) * 2.0);

    weight_vector &correct_weight = weight_[datum.category];
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      if (correct_weight.size() <= it->first)
        correct_weight.resize(it->first + 1, 0.0);
      correct_weight[it->first] += step_distance * it->second;
    }

    if (non_correct_predict == non_class)
      return;

    weight_vector &wrong_weight = weight_[non_correct_predict];
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      if (wrong_weight.size() <= it->first)
        wrong_weight.resize(it->first + 1, 0.0);

      wrong_weight[it->first] -= step_distance * it->second;
    }
  }
}

void FOBOS::CalcScores(const feature_vector& fv,
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

void FOBOS::GetFeatureWeight(size_t feature_id,
                             std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature_id, weight_, results);
}

} //namespace
} //namespace
