#ifndef CLASSIFIER_TOOL_WEIGHT_H_
#define CLASSIFIER_TOOL_WEIGHT_H_

#include <unordered_map>
namespace classifier {
typedef std::unordered_map<std::string, double> weight_vector;
typedef std::unordered_map<std::string, weight_vector> weight_matrix;
} //namespace

#endif //CLASSIFIER_TOOL_WEIGHT_H_
