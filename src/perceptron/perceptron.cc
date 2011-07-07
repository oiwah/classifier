#include <perceptron/perceptron.h>

#include <algorithm>

namespace classifier {
namespace perceptron {
Perceptron::Perceptron() {
  weight_matrix().swap(weight_);
}

void Perceptron::Train(const datum& datum) {
  std::string predict;
  Test(datum.fv, &predict);
  Update(datum.fv, datum.category, predict);
}

void Perceptron::Train(const std::vector<datum>& data,
                       const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      Train(data[i]);
    }
  }
}

void Perceptron::Test(const feature_vector& fv,
                      std::string* predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  CalcScores(fv, &score2class);
  *predict = score2class[0].second;
}

void Perceptron::CalcScores(const feature_vector& fv,
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

void Perceptron::GetFeatureWeight(const std::string& feature,
                                  std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature, weight_, results);
}

} //namespace
} //namespace
