#include <sstream>

#include <tool/feature.h>

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
