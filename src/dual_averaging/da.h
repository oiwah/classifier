#ifndef CLASSIFIER_DUAL_AVERAGING_DA_H_
#define CLASSIFIER_DUAL_AVERAGING_DA_H_

#include <iostream>
#include <vector>

#include <tool/calc.h>

namespace classifier {
namespace dual_averaging {
class DualAveraging {
 public:
  explicit DualAveraging(double gamma = 1.0);
  ~DualAveraging() {};

  void Train(const datum& datum, bool primal = true);
  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv, std::string* predict) const;
  void GetFeatureWeight(const std::string& feature,
                        std::vector<std::pair<std::string, double> >* results) const;

 private:
  void CalcWeight(const feature_vector& fv);
  void CalcWeightAll();

  void CalcScores(const feature_vector& fv,
                  score2class* scores) const;

  void Update(const datum& datum,
              const score2class& scores);

  weight_matrix weight_;
  weight_matrix subgradient_sum_;
  size_t dataN_;
  double gamma_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_DUAL_AVERAGING_DA_H_
