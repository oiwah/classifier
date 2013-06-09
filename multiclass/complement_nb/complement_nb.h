#ifndef CLASSIFIER_COMPLEMENTNB_COMPLEMENTNB_H_
#define CLASSIFIER_COMPLEMENTNB_COMPLEMENTNB_H_

#include "../naivebayes/nb.h"

namespace classifier {
namespace naivebayes {

class ComplementNaiveBayes : public NaiveBayes {
 public:
  ~ComplementNaiveBayes() {};

 private:
  double CalculateProbability(const feature_vector& fv,
                              const std::string& category) const;

};

} //namespace
} //namespace

#endif //CLASSIFIER_COMPLEMENTNB_COMPLEMENTNB_H_
