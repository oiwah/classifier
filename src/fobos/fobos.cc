#include <fobos/fobos.h>

#include <cmath>
#include <algorithm>

namespace classifier {
namespace fobos {
FOBOS::FOBOS(double eta, double lambda) : dataN_(0), eta_(eta), lambda_(lambda), truncate_sum_(0.0) {
  weight_matrix().swap(weight_);
  weight_matrix().swap(prev_truncate_);
}

void FOBOS::Train(const datum& datum) {
  std::string non_correct_predict;
  double hinge_loss = CalcHingeLoss(datum, &non_correct_predict);
  Update(datum.category, non_correct_predict, hinge_loss, datum.fv);
  TruncateAll();
}

void FOBOS::Train(const std::vector<datum>& data,
                  const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      Train(data[i]);
    }
  }
  TruncateAll();
}

void FOBOS::Test(const feature_vector& fv,
                 std::string* predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  CalcScores(fv, &score2class);
  *predict = score2class[0].second;
}

void FOBOS::Truncate(const std::string correct,
                     const std::string non_correct_predict,
                     const feature_vector& fv) {
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    std::map<std::string, double> correct_prev = prev_truncate_[correct];
    if (correct_prev.find(it->first) != correct_prev.end()) {
      double truncate_value = truncate_sum_ - correct_prev[it->first];
      if (weight_[correct][it->first] > 0.0) {
        weight_[correct][it->first]
            = std::max(0.0,
                       weight_[correct][it->first] - truncate_value);
      } else {
        weight_[correct][it->first]
            = std::min(0.0,
                       weight_[correct][it->first] + truncate_value);
      }
    }

    prev_truncate_[correct][it->first] = truncate_sum_;

    std::map<std::string, double> predict_prev = prev_truncate_[non_correct_predict];
    if (predict_prev.find(it->first) != predict_prev.end()) {
      double truncate_value = truncate_sum_ + predict_prev[it->first];
      if (weight_[non_correct_predict][it->first] > 0.0) {
        weight_[non_correct_predict][it->first]
            = std::max(0.0,
                       weight_[non_correct_predict][it->first] - truncate_value);
      } else {
        weight_[non_correct_predict][it->first]
            = std::min(0.0,
                       weight_[non_correct_predict][it->first] + truncate_value);
      }
    }

    prev_truncate_[non_correct_predict][it->first] = truncate_sum_;
  }
}

void FOBOS::TruncateAll() {
  for (weight_matrix::const_iterator wm_it = weight_.begin();
       wm_it != weight_.end();
       ++wm_it) {
    weight_vector wv = wm_it->second;
    for (weight_vector::const_iterator wv_it = wv.begin();
         wv_it != wv.end();
         ++wv_it) {
      double prev_truncate = prev_truncate_[wm_it->first][wv_it->first];
      double truncate_value = truncate_sum_ - prev_truncate;
      if (wv_it->second > 0.0) {
        weight_[wm_it->first][wv_it->first]
            = std::max(0.0, wv_it->second - truncate_value);
      } else {
        weight_[wm_it->first][wv_it->first]
            = std::min(0.0, wv_it->second + truncate_value);
      }
      prev_truncate_[wv_it->first][wv_it->first] = truncate_sum_;
    }
  }
}

void FOBOS::CalcScores(const feature_vector& fv,
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

void FOBOS::Update(const std::string& correct,
                   const std::string& non_correct_predict,
                   const double hinge_loss,
                   const feature_vector& fv) {
  Truncate(correct, non_correct_predict, fv);
  ++dataN_;
  truncate_sum_ += lambda_ * eta_ / (std::sqrt(dataN_) * 2.0);

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

double FOBOS::CalcHingeLoss(const datum& datum,
                            std::string* non_correct_predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  CalcScores(datum.fv, &score2class);

  bool correct_done = false;
  bool predict_done = false;
  double score = 1.0;
  for (std::vector<std::pair<double, std::string> >::const_iterator
           it = score2class.begin();
       it != score2class.end();
       ++it) {
    if (it->second == datum.category) {
      score -= it->first;
      correct_done = true;
    } else if (!predict_done) {
      *non_correct_predict = it->second;
      if (*non_correct_predict != non_class)
        score += it->first;
      predict_done = true;
    }

    if (correct_done && predict_done)
      break;
  }

  return score;
}

void FOBOS::GetFeatureWeight(const std::string& feature,
                             std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature, weight_, results);
}

} //namespace
} //namespace
