#ifndef CLASSIFIER_TOOL_CALC_H_
#define CLASSIFIER_TOOL_CALC_H_

#include <tool/feature.h>
#include <tool/weight.h>

namespace classifier {
typedef std::vector<std::pair<double, std::string> > score2class;

inline double InnerProduct(const feature_vector& fv,
                           weight_vector* wv) {
  double score = 0.0;
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it)
    score += (*wv)[it->first] * it->second;
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

inline void ReturnFeatureWeight(const std::string& feature,
                                const weight_matrix& wm,
                                std::vector<std::pair<std::string, double> >* results) {
  for (weight_matrix::const_iterator it = wm.begin();
       it != wm.end();
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
} //namespace

#endif //CLASSIFIER_TOOL_CALC_H_
