#ifndef CLASSIFIER_UTILITY_LIBSVMPARSER_H_
#define CLASSIFIER_UTILITY_LIBSVMPARSER_H_

#include <sstream>

#include "feature.h"

namespace classifier {
namespace parser {
void LibsvmParser(std::istringstream* iss,
                  classifier::datum* datum) {
  size_t id = 0;
  char comma = 0;
  double value = 0.0;
  while (*iss >> id >> comma >> value)
    datum->fv.push_back(std::make_pair(id, value));
}
} //namespace parser
} //namespace classifier

#endif //CLASSIFIER_UTILITY_LIBSVMPARSER_H_
