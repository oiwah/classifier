#include <perceptron/averaged_perceptron.h>

#include <algorithm>

namespace classifier {
namespace perceptron {
AveragedPerceptron::AveragedPerceptron() : dataN_(0) {
  weight_matrix().swap(weight_);
  weight_matrix().swap(differential_weight_);
  weight_matrix().swap(averaged_weight_);
}

void AveragedPerceptron::Train(const datum& datum,
                               const bool calc_averaged) {
  ++dataN_;
  std::string predict;
  Predict(datum.fv, &predict);
  Update(datum, predict);

  if (calc_averaged)
    CalcAveragedWeight();
}

void AveragedPerceptron::Train(const std::vector<datum>& data,
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

void AveragedPerceptron::Test(const feature_vector& fv,
                              std::string* predict) const {
  Predict(fv, predict, 1);
}

void AveragedPerceptron::Predict(const feature_vector& fv,
                                 std::string* predict,
                                 size_t mode) const {
  score2class scores(0);
  scores.push_back(make_pair(non_class_score, non_class));

  if (mode == 0) {
    for (weight_matrix::const_iterator it = weight_.begin();
         it != weight_.end();
         ++it) {
      double score = InnerProduct(fv, it->second);
      scores.push_back(make_pair(score, it->first));
    }
  } else if (mode == 1) {
    for (weight_matrix::const_iterator it = averaged_weight_.begin();
         it != averaged_weight_.end();
         ++it) {
      double score = InnerProduct(fv, it->second);
      scores.push_back(make_pair(score, it->first));
    }
  }

  sort(scores.begin(), scores.end(),
       std::greater<std::pair<double, std::string> >());
  *predict = scores[0].second;
}

void AveragedPerceptron::Update(const datum& datum,
                                const std::string& predict) {
  if (datum.category == predict)
    return;

  weight_vector &correct_weight = weight_[datum.category];
  weight_vector &correct_diffetial_weight = differential_weight_[datum.category];
  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    if (correct_weight.size() <= it->first)
      correct_weight.resize(it->first + 1, 0.0);
    correct_weight[it->first] += it->second / 2.0;

    if (correct_diffetial_weight.size() <= it->first)
      correct_diffetial_weight.resize(it->first + 1, 0.0);
    correct_diffetial_weight[it->first] += dataN_ * it->second / 2.0;
  }

  if (predict == non_class)
    return;

  weight_vector &wrong_weight = weight_[predict];
  weight_vector &wrong_diffetial_weight = differential_weight_[predict];
  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    if (wrong_weight.size() <= it->first)
      wrong_weight.resize(it->first + 1, 0.0);
    wrong_weight[it->first] -= it->second / 2.0;

    if (wrong_diffetial_weight.size() <= it->first)
      wrong_diffetial_weight.resize(it->first + 1, 0.0);
    wrong_diffetial_weight[it->first] -= dataN_ * it->second / 2.0;
  }
}

void AveragedPerceptron::CalcAveragedWeight() {
  weight_matrix ave_wm;

  for (weight_matrix::const_iterator wm_it = weight_.begin();
       wm_it != weight_.end();
       ++wm_it) {
    weight_vector diff_wv = differential_weight_[wm_it->first];
    weight_vector wv = wm_it->second;

    weight_vector &ave_wv = ave_wm[wm_it->first];
    ave_wv.resize(wv.size(), 0.0);

    for (size_t feature_id = 0; feature_id < wv.size(); ++feature_id) {
      ave_wv[feature_id] = wv[feature_id] - diff_wv[feature_id] / dataN_;
    }
  }

  averaged_weight_.swap(ave_wm);
}

void AveragedPerceptron::GetFeatureWeight(size_t feature_id,
                                          std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature_id, averaged_weight_, results);
}

} //namespace
} //namespace
