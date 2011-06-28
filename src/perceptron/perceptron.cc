#include <perceptron/perceptron.h>

#include <cfloat>
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

      if (predict == data[i].category) {
        continue;
      } else if (predict != non_class) {
        Update(predict, data[i].fv, -1.0);
      }

      Update(data[i].category, data[i].fv);
    }
  }
}

void Perceptron::Test(const feature_vector& fv,
                      std::string* predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  score2class.push_back(make_pair(-DBL_MAX, non_class));

  for (weight_matrix::const_iterator it = weight_.begin();
       it != weight_.end();
       ++it) {
    weight_vector wv = it->second;
    double score = CalcScore(fv, wv);
    score2class.push_back(make_pair(score, it->first));
  }

  sort(score2class.begin(), score2class.end(),
       std::greater<std::pair<double, std::string> >());

  *predict = score2class[0].second;
}

double Perceptron::CalcScore(const feature_vector& fv,
                             weight_vector weight_vec) const {
  double score = 0.0;
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it)
    score += weight_vec[it->first] * it->second;

  return score;
}

void Perceptron::Update(const std::string& category,
                        const feature_vector& fv,
                        const double eta) {
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it)
    weight_[category][it->first] += eta * it->second;
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
