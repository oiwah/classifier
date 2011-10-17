#ifndef CLASSIFIER_LOGLINEAR_LOGLINER_SGD_H_
#define CLASSIFIER_LOGLINEAR_LOGLINER_SGD_H_

#include <vector>
#include <iostream>

#include <tool/calc.h>

namespace classifier {
namespace loglinear {

class LogLinearSGD {
 public:
  explicit LogLinearSGD(double eta = 1.0);
  ~LogLinearSGD() {};

  void Train(const datum& datum);
  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv, std::string* predict) const;
  void GetFeatureWeight(size_t feature_id,
                        std::vector<std::pair<std::string, double> >* results) const;
 private:
  void CalcScores(const feature_vector& fv,
                  score2class* scores) const;

  void Update(const datum& datum,
              const score2class& scores);

  double eta_;
  weight_matrix weight_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_LOGLINEAR_LOGLINEAR_SGD_H_
