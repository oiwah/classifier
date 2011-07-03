#include <perceptron/averaged_perceptron.h>

#include <algorithm>

namespace classifier {
namespace perceptron {
AveragedPerceptron::AveragedPerceptron() : dataN_(0) {
  weight_matrix().swap(weight_);
  weight_matrix().swap(differential_weight_);
  weight_matrix().swap(averaged_weight_);
}

void AveragedPerceptron::Train(const std::vector<datum>& data,
                               const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      ++dataN_;
      std::string predict;
      Predict(false, data[i].fv, &predict);
      Update(data[i].fv, data[i].category, predict);
    }
  }
  CalcAveragedWeight();
}

void AveragedPerceptron::Test(const feature_vector& fv,
                              std::string* predict) const {
  Predict(true, fv, predict);
}

void AveragedPerceptron::Predict(bool averaged,
                                 const feature_vector& fv,
                                 std::string* predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  score2class.push_back(make_pair(non_class_score, non_class));

  if (averaged) {
    for (weight_matrix::const_iterator it = averaged_weight_.begin();
         it != averaged_weight_.end();
         ++it) {
      weight_vector wv = it->second;
      double score = InnerProduct(fv, &wv);
      score2class.push_back(make_pair(score, it->first));
    }
  } else {
    for (weight_matrix::const_iterator it = weight_.begin();
         it != weight_.end();
         ++it) {
      weight_vector wv = it->second;
      double score = InnerProduct(fv, &wv);
      score2class.push_back(make_pair(score, it->first));
    }
  }
  sort(score2class.begin(), score2class.end(),
       std::greater<std::pair<double, std::string> >());
  *predict = score2class[0].second;
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

void AveragedPerceptron::CompareFeatureWeight(const std::string& feature,
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
