#ifndef CLASSIFIER_TOOL_CALC_H_
#define CLASSIFIER_TOOL_CALC_H_

#include "feature.h"
#include "weight.h"

namespace classifier {
typedef std::vector<std::pair<double, std::string> > score2class;
inline double InnerProduct(const feature_vector& fv,
                           const weight_vector& wv) {
  double score = 0.0;
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    if (wv.size() <= it->first) continue;
    score += wv[it->first] * it->second;
  }
  return score;
}

inline double CalcFvNorm(const feature_vector& fv) {
  double fv_norm = 0.0;
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it)
    fv_norm += it->second * it->second;

  return fv_norm;
}

inline void ReturnFeatureWeight(size_t feature_id,
                                const weight_matrix& wm,
                                std::vector<std::pair<std::string, double> >* results) {
  for (weight_matrix::const_iterator it = wm.begin();
       it != wm.end();
       ++it) {
    std::string category = it->first;
    if (feature_id < it->second.size()) {
      double score = it->second.at(feature_id);
      results->push_back(make_pair(category, score));
    } else {
      results->push_back(make_pair(category, 0.0));
    }
  }
}

inline double CalcLossScore(const score2class& s2c,
                            const std::string& correct,
                            std::string* non_correct_predict,
                            const double margin = 0.0) {
  bool correct_done = false;
  bool predict_done = false;
  double loss_score = margin;

  for (score2class::const_iterator it = s2c.begin();
       it != s2c.end();
       ++it) {
    if (it->second == correct) {
      loss_score -= it->first;
      correct_done = true;
    } else if (!predict_done) {
      *non_correct_predict = it->second;
      if (*non_correct_predict != non_class)
        loss_score += it->first;
      predict_done = true;
    }

    if (correct_done && predict_done)
      break;
  }

  return loss_score;
}

inline double CalcLossScore(const double score,
                            const int correct,
                            const double margin = 0.0) {
  if (correct == 1) {
    return margin - score;
  }
  return score - margin;
}
} //namespace

#endif //CLASSIFIER_TOOL_CALC_H_
