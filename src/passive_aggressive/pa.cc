#include <passive_aggressive/pa.h>

#include <cfloat>
#include <algorithm>

namespace classifier {
namespace pa {
PA::PA(size_t mode) : mode_(mode), C_(0.001) {
  weight_matrix().swap(weight_);
}

void PA::SetC(double C) {
  if (C > 0.0)
    C_ = C;
}

void PA::Train(const std::vector<datum>& data,
               const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (size_t i = 0; i < data.size(); ++i) {
      std::vector<std::pair<double, std::string> > score2class(0);
      CalcScores(data[i].fv, &score2class);

      Update(data[i].category, score2class, data[i].fv);
    }
  }
}

void PA::Test(const feature_vector& fv,
              std::string* predict) const {
  std::vector<std::pair<double, std::string> > score2class(0);
  CalcScores(fv, &score2class);
  *predict = score2class[0].second;
}

void PA::CalcScores(const feature_vector& fv,
                    std::vector<std::pair<double, std::string> >* score2class) const {
  score2class->push_back(make_pair(-DBL_MAX, non_class));

  for (weight_matrix::const_iterator it = weight_.begin();
       it != weight_.end();
       ++it) {
    weight_vector wv = it->second;
    double score = InnerProduct(fv, wv);
    score2class->push_back(make_pair(score, it->first));
  }

  sort(score2class->begin(), score2class->end(),
       std::greater<std::pair<double, std::string> >());
}

double PA::InnerProduct(const feature_vector& fv,
                        weight_vector& wv) const {
  double score = 0.0;
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it)
    score += wv[it->first] * it->second;

  return score;
}

double PA::CalcHingeLoss(const std::vector<std::pair<double, std::string> >& score2class,
                         const std::string& correct,
                         std::string* non_correct_predict) const {
  bool correct_done = false;
  bool predict_done = false;
  double score = 1.0;
  for (std::vector<std::pair<double, std::string> >::const_iterator
           it = score2class.begin();
       it != score2class.end();
       ++it) {
    if (it->second == correct) {
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

double PA::CalcFvNorm(const feature_vector& fv) const {
  double fv_norm = 0.0;
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it)
    fv_norm += it->second * it->second;

  return fv_norm;
}

void PA::Update(const std::string& correct,
                const std::vector<std::pair<double, std::string> >& score2class,
                const feature_vector& fv) {
  std::string non_correct_predict;
  double hinge_loss = CalcHingeLoss(score2class, correct, &non_correct_predict);
  double fv_norm = CalcFvNorm(fv);
  double update = 0.0;

  switch(mode_) {
    case 0:
      update = hinge_loss / (fv_norm * 2.0);
      break;

    case 1:
      update = std::min(hinge_loss / (fv_norm * 2.0), C_);
      break;

    case 2:
      update = hinge_loss / (fv_norm + 1 / (2.0 * C_) * 2.0);
      break;

    default:
      break;
  }

  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    weight_[correct][it->first] += update;
    weight_[non_correct_predict][it->first] -= update;
  }
}

void PA::CompareFeatureWeight(const std::string& feature,
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
