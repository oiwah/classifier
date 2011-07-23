#include <subgradient/hinge.h>

#include <cmath>
#include <algorithm>

namespace classifier {
namespace subgradient {
SubgradientHinge::SubgradientHinge(double eta) : dataN_(0), eta_(eta) {
  weight_matrix().swap(weight_);
}

void SubgradientHinge::Train(const datum& datum) {
  ++dataN_;
  score2class scores(0);
  CalcScores(datum.fv, &scores);

  Update(datum.category, scores, datum.fv);
}

void SubgradientHinge::Train(const std::vector<datum>& data,
                             const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      Train(data[i]);
    }
  }
}

void SubgradientHinge::Test(const feature_vector& fv,
                            std::string* predict) const {
  score2class scores(0);
  CalcScores(fv, &scores);
  *predict = scores[0].second;
}

void SubgradientHinge::CalcScores(const feature_vector& fv,
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

void SubgradientHinge::Update(const std::string& correct,
                              const score2class& scores,
                              const feature_vector& fv) {
  std::string non_correct_predict;
  double hinge_loss = CalcLossScore(scores, correct, &non_correct_predict, 1.0);

  if (hinge_loss > 0.0) {
    double step_distance = eta_ / (std::sqrt(dataN_) * 2.0);

    for (feature_vector::const_iterator it = fv.begin();
         it != fv.end();
         ++it) {
      weight_[correct][it->first] += step_distance * it->second;
      if (non_correct_predict != non_class)
        weight_[non_correct_predict][it->first] -= step_distance * it->second;
    }
  }
}

void SubgradientHinge::GetFeatureWeight(const std::string& feature,
                                        std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature, weight_, results);
}

} //namespace
} //namespace
