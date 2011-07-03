#ifndef CLASSIFIER_TOOL_CALC_H_
#define CLASSIFIER_TOOL_CALC_H_

#include <tool/feature.h>
#include <tool/weight.h>

namespace classifier {
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


} //namespace

#endif //CLASSIFIER_TOOL_CALC_H_
