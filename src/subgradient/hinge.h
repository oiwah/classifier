#ifndef CLASSIFIER_SUBGRADIENT_HINGE_H_
#define CLASSIFIER_SUBGRADIENT_HINGE_H_

#include <iostream>
#include <vector>

#include <tool/calc.h>

namespace classifier {
namespace subgradient {
class SubgradientHinge {
 public:
  explicit SubgradientHinge(double eta = 1.0);
  ~SubgradientHinge() {};

  void Train(const datum& datum);
  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv, std::string* predict) const;
  void GetFeatureWeight(const std::string& feature,
                        std::vector<std::pair<std::string, double> >* results) const;

 private:
  void CalcScores(const feature_vector& fv,
                  std::vector<std::pair<double, std::string> >* score2class) const;

  void Update(const std::string& correct,
              const std::vector<std::pair<double, std::string> >& score2class,
              const feature_vector& fv);

  double CalcHingeLoss(const std::vector<std::pair<double, std::string> >& score2class,
                       const std::string& correct,
                       std::string* non_correct_predict) const;

  weight_matrix weight_;
  size_t dataN_;
  double eta_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_SUBGRADIENT_HINGE_H_
