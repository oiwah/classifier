#include <loglinear/loglinear_sgd.h>

#include <cmath>
#include <algorithm>

namespace classifier {
namespace loglinear {
LogLinearSGD::LogLinearSGD(double eta) : eta_(eta) {
  weight_matrix().swap(weight_);
}

void LogLinearSGD::Train(const datum& datum) {
  score2class scores(0);
  CalcScores(datum.fv, &scores);
  Update(datum, scores);
}

void LogLinearSGD::Train(const std::vector<datum>& data,
                         const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (std::vector<datum>::const_iterator it = data.begin();
         it != data.end();
         ++it) {
      Train(*it);
    }
  }
}

void LogLinearSGD::Test(const feature_vector& fv,
                        std::string* predict) const {
  score2class scores(0);
  CalcScores(fv, &scores);
  *predict = scores[0].second;
}

void LogLinearSGD::CalcScores(const feature_vector& fv,
                              score2class* scores) const {
  double max_score = 0.0;
  for (weight_matrix::const_iterator it = weight_.begin();
       it != weight_.end();
       ++it) {
    double score = InnerProduct(fv, it->second);
    max_score = std::max(max_score, score);
    scores->push_back(make_pair(score, it->first));
  }

  double sum_score = 0.0;
  for (score2class::iterator it = scores->begin();
       it != scores->end();
       ++it) {
    it->first = std::exp(it->first - max_score);
    sum_score += it->first;
  }

  for (score2class::iterator it = scores->begin();
       it != scores->end();
       ++it) {
    it->first /= sum_score;
  }

  sort(scores->begin(), scores->end(),
       std::greater<std::pair<double, std::string> >());
}

void LogLinearSGD::Update(const datum& datum,
                          const score2class& scores) {
  weight_vector &correct_weight = weight_[datum.category];
  for (feature_vector::const_iterator fv_it = datum.fv.begin();
       fv_it != datum.fv.end();
       ++fv_it) {
    for (size_t i = 0; i < scores.size(); ++i) {
      weight_vector &weight_vec = weight_[scores[i].second];
      if (weight_vec.size() <= fv_it->first)
        weight_vec.resize(fv_it->first + 1, 0.0);
      weight_vec[fv_it->first] -= eta_ * scores[i].first * fv_it->second;
    }

    if (correct_weight.size() <= fv_it->first)
      correct_weight.resize(fv_it->first + 1, 0.0);

    correct_weight[fv_it->first] += eta_ * fv_it->second;
  }
}

void LogLinearSGD::GetFeatureWeight(size_t feature_id,
                                    std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature_id, weight_, results);
}

} //namespace
} //namespace
