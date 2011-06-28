#ifndef CLASSIFIER_TOOL_FEATURE_H_
#define CLASSIFIER_TOOL_FEATURE_H_

namespace classifier {
typedef std::map<std::string, double> weight_vector;
typedef std::map<std::string, weight_vector> weight_matrix;

typedef std::map<std::string, double> feature_vector;
struct datum {
  std::string category;
  feature_vector fv;
};
} //namespace

#endif //CLASSIFIER_TOOL_FEATURE_H_

