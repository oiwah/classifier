#include "averaged_hinge.h"

#include <cmath>
#include <algorithm>

namespace classifier {
namespace subgradient {
ASGDHinge::ASGDHinge(double eta) : dataN_(0), eta_(eta) {
  weight_matrix().swap(weight_);
  weight_matrix().swap(differential_weight_);
  weight_matrix().swap(averaged_weight_);
}

void ASGDHinge::Train(const datum& datum,
                      bool calc_averaged) {
  ++dataN_;
  score2class scores(0);
  CalcScores(datum.fv, &scores, 0);
  Update(datum, scores);

  if (calc_averaged)
    CalcAveragedWeight();
}

void ASGDHinge::Train(const std::vector<datum>& data,
                      const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (std::vector<datum>::const_iterator it = data.begin();
         it != data.end();
         ++it) {
      Train(*it, false);
    }
  }

  CalcAveragedWeight();
}

void ASGDHinge::Test(const feature_vector& fv,
                     std::string* predict) const {
  score2class scores(0);
  CalcScores(fv, &scores, 1);
  *predict = scores[0].second;
}

void ASGDHinge::CalcScores(const feature_vector& fv,
                           score2class* scores,
                           size_t mode) const {
  scores->push_back(make_pair(non_class_score, non_class));

  if (mode == 0) {
    for (weight_matrix::const_iterator it = weight_.begin();
         it != weight_.end();
         ++it) {
      double score = InnerProduct(fv, it->second);
      scores->push_back(make_pair(score, it->first));
    }
  } else if (mode == 1) {
    for (weight_matrix::const_iterator it = averaged_weight_.begin();
         it != averaged_weight_.end();
         ++it) {
      double score = InnerProduct(fv, it->second);
      scores->push_back(make_pair(score, it->first));
    }
  }

  sort(scores->begin(), scores->end(),
       std::greater<std::pair<double, std::string> >());
}

void ASGDHinge::Update(const datum& datum,
                       const score2class& scores) {
  std::string non_correct_predict;
  double hinge_loss = CalcLossScore(scores, datum.category, &non_correct_predict, 1.0);

  if (hinge_loss > 0.0) {
    double step_distance = eta_ / (std::sqrt(dataN_) * 2.0);

    weight_vector &correct_weight = weight_[datum.category];
    weight_vector &correct_diffetial_weight = differential_weight_[datum.category];
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      if (correct_weight.size() <= it->first)
        correct_weight.resize(it->first + 1, 1.0);
      correct_weight[it->first] += step_distance * it->second;

      if (correct_diffetial_weight.size() <= it->first)
        correct_diffetial_weight.resize(it->first + 1, 0.0);
      correct_diffetial_weight[it->first] += dataN_ * step_distance * it->second;
    }

    if (non_correct_predict == non_class)
      return;

    weight_vector &wrong_weight = weight_[non_correct_predict];
    weight_vector &wrong_diffetial_weight = differential_weight_[non_correct_predict];
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      if (wrong_weight.size() <= it->first)
        wrong_weight.resize(it->first + 1, 1.0);
      wrong_weight[it->first] -= step_distance * it->second;

      if (wrong_diffetial_weight.size() <= it->first)
        wrong_diffetial_weight.resize(it->first + 1, 0.0);
      wrong_diffetial_weight[it->first] -= dataN_ * step_distance * it->second;
    }
  }
}

void ASGDHinge::CalcAveragedWeight() {
  weight_matrix ave_wm;

  for (weight_matrix::const_iterator wm_it = weight_.begin();
       wm_it != weight_.end();
       ++wm_it) {
    weight_vector &diff_wv = differential_weight_[wm_it->first];
    weight_vector &wv = weight_[wm_it->first];

    weight_vector &ave_wv = ave_wm[wm_it->first];
    ave_wv.resize(wv.size(), 0.0);

    for (size_t feature_id = 0; feature_id < wv.size(); ++feature_id) {
      ave_wv[feature_id] = wv[feature_id] - diff_wv[feature_id] / dataN_;
    }
  }

  averaged_weight_.swap(ave_wm);
}

void ASGDHinge::GetFeatureWeight(size_t feature_id,
                                 std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature_id, weight_, results);
}

} //namespace
} //namespace
