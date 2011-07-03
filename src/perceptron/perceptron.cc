#include <perceptron/perceptron.h>

#include <algorithm>

namespace classifier {
namespace perceptron {
Perceptron::Perceptron() {
  weight_matrix().swap(weight_);
}

void Perceptron::Train(const std::vector<datum>& data,
                       const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      std::string predict;
      Test(data[i].fv, &predict);
      Update(data[i].fv, data[i].category, predict);
    }
  }
}

void Perceptron::Test(const feature_vector& fv,
                      std::string* predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  score2class.push_back(make_pair(non_class_score, non_class));

  for (weight_matrix::const_iterator it = weight_.begin();
       it != weight_.end();
       ++it) {
    weight_vector wv = it->second;
    double score = InnerProduct(fv, &wv);
    score2class.push_back(make_pair(score, it->first));
  }

  sort(score2class.begin(), score2class.end(),
       std::greater<std::pair<double, std::string> >());

  *predict = score2class[0].second;
}

void Perceptron::Update(const feature_vector& fv,
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
    }
  } else {
    for (feature_vector::const_iterator it = fv.begin();
         it != fv.end();
         ++it) {
      weight_[correct][it->first] += it->second;
    }
  }
}

void Perceptron::CompareFeatureWeight(const std::string& feature,
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
