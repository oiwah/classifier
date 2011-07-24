#include <dual_averaging/da.h>

#include <cmath>
#include <algorithm>

namespace classifier {
namespace dual_averaging {
DualAveraging::DualAveraging(double gamma) : dataN_(0), gamma_(gamma) {
  weight_matrix().swap(weight_);
  weight_matrix().swap(subgradient_sum_);
}

void DualAveraging::Train(const datum& datum,
                          bool primal) {
  CalcWeight(datum.fv);

  ++dataN_;
  score2class scores(0);
  CalcScores(datum.fv, &scores);
  Update(datum, scores);

  if (primal)
    CalcWeightAll();
}

void DualAveraging::Train(const std::vector<datum>& data,
                          const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      Train(data[i], false);
    }
  }
  CalcWeightAll();
}

void DualAveraging::Test(const feature_vector& fv,
                         std::string* predict) const {
  score2class scores(0);
  CalcScores(fv, &scores);
  *predict = scores[0].second;
}

void DualAveraging::CalcWeight(const feature_vector& fv) {
  double scalar = 0.0;
  if (dataN_ != 0) scalar = - 1.0 / sqrt(dataN_) * gamma_;

  for (feature_vector::const_iterator fv_it = fv.begin();
       fv_it != fv.end();
       ++fv_it) {
    for (weight_matrix::const_iterator wm_it = subgradient_sum_.begin();
         wm_it != subgradient_sum_.end();
         ++wm_it) {
      weight_[wm_it->first][fv_it->first]
          = scalar * subgradient_sum_[wm_it->first][fv_it->first];
    }
  }
}

void DualAveraging::CalcWeightAll() {
  double scalar = - sqrt(dataN_) / gamma_;
  for (weight_matrix::const_iterator wm_it = subgradient_sum_.begin();
       wm_it != subgradient_sum_.end();
       ++wm_it) {
    weight_vector wv = wm_it->second;
    for (weight_vector::const_iterator wv_it = wv.begin();
         wv_it != wv.end();
         ++wv_it) {
      weight_[wm_it->first][wv_it->first]
          = scalar * subgradient_sum_[wm_it->first][wv_it->first];
    }
  }
}

void DualAveraging::CalcScores(const feature_vector& fv,
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

void DualAveraging::Update(const datum& datum,
                           const score2class& scores) {
  std::string non_correct_predict;
  double hinge_loss = CalcLossScore(scores, datum.category, &non_correct_predict, 1.0);

  if (hinge_loss > 0.0) {
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      subgradient_sum_[datum.category][it->first] -= it->second / 2.0;
      if (non_correct_predict == non_class) continue;
      subgradient_sum_[non_correct_predict][it->first] += it->second / 2.0;
    }
  }
}

void DualAveraging::GetFeatureWeight(const std::string& feature,
                                     std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature, weight_, results);
}

} //namespace
} //namespace
