#ifndef CLASSIFIER_TOOL_FEATURE_H_
#define CLASSIFIER_TOOL_FEATURE_H_

#include <cfloat>
namespace classifier {
typedef std::map<std::string, double> feature_vector;
struct datum {
  std::string category;
  feature_vector fv;
};

const std::string non_class = "None";
const double non_class_score = -DBL_MAX;
} //namespace

#endif //CLASSIFIER_TOOL_FEATURE_H_

