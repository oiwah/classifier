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
  Predict(datum.fv, weight_, &predict);
  Update(datum.fv, datum.category, predict);

  if (calc_averaged)
    CalcAveragedWeight();
}

void AveragedPerceptron::Train(const std::vector<datum>& data,
                               const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      Train(data[i], false);
    }
  }
  CalcAveragedWeight();
}

void AveragedPerceptron::Test(const feature_vector& fv,
                              std::string* predict) const {
  Predict(fv, averaged_weight_, predict);
}

void AveragedPerceptron::Predict(const feature_vector& fv,
                                 const weight_matrix& wm,
                                 std::string* predict) const {
  score2class scores(0);
  scores.push_back(make_pair(non_class_score, non_class));
  for (weight_matrix::const_iterator it = wm.begin();
       it != wm.end();
       ++it) {
    weight_vector wv = it->second;
    double score = InnerProduct(fv, &wv);
    scores.push_back(make_pair(score, it->first));
  }
  sort(scores.begin(), scores.end(),
       std::greater<std::pair<double, std::string> >());
  *predict = scores[0].second;
}

void AveragedPerceptron::Update(const feature_vector& fv,
                                const std::string& correct,
                                const std::string& predict) {
  if (correct == predict)
    return;

  if (predict != non_class) {
    for (feature_vector::const_iterator it = fv.begin();
         it != fv.end();
         ++it) {
      weight_[correct][it->first] += it->second;
      weight_[predict][it->first] -= it->second;
      differential_weight_[correct][it->first] += dataN_ * it->second;
      differential_weight_[predict][it->first] -= dataN_ * it->second;
    }
  } else {
    for (feature_vector::const_iterator it = fv.begin();
         it != fv.end();
         ++it) {
      weight_[correct][it->first] += it->second;
      differential_weight_[correct][it->first] += dataN_ * it->second;
    }
  }
}

void AveragedPerceptron::CalcAveragedWeight() {
  weight_matrix wm;

  for (weight_matrix::const_iterator wm_it = weight_.begin();
       wm_it != weight_.end();
       ++wm_it) {
    weight_vector wv = wm_it->second;
    for (weight_vector::const_iterator wv_it = wv.begin();
         wv_it != wv.end();
         ++wv_it) {
      wm[wm_it->first][wv_it->first] = wv_it->second
          - differential_weight_[wm_it->first][wv_it->first] / dataN_;
    }
  }

  averaged_weight_.swap(wm);
}

void AveragedPerceptron::GetFeatureWeight(const std::string& feature,
                                          std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature, averaged_weight_, results);
}

} //namespace
} //namespace
