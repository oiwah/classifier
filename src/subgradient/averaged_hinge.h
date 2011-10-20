#ifndef CLASSIFIER_SUBGRADIENT_AVERAGED_HINGE_H_
#define CLASSIFIER_SUBGRADIENT_AVERAGED_HINGE_H_

#include <iostream>
#include <vector>

#include <tool/calc.h>

namespace classifier {
namespace subgradient {
class ASGDHinge {
 public:
  explicit ASGDHinge(double eta = 1.0);
  ~ASGDHinge() {};

  void Train(const datum& datum,
             bool calc_averaged = true);
  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv, std::string* predict) const;
  void GetFeatureWeight(size_t feature_id,
                        std::vector<std::pair<std::string, double> >* results) const;

 private:
  void CalcScores(const feature_vector& fv,
                  score2class* scores,
                  size_t mode) const;

  void Update(const datum& datum,
              const score2class& scores);

  void CalcAveragedWeight();

  weight_matrix weight_;
  weight_matrix differential_weight_;
  weight_matrix averaged_weight_;

  size_t dataN_;
  double eta_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_SUBGRADIENT_AVERAGED_HINGE_H_
