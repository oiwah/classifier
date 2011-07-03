#ifndef CLASSIFIER_PASSIVE_AGGRESSIVE_PA_H_
#define CLASSIFIER_PASSIVE_AGGRESSIVE_PA_H_

#include <iostream>
#include <vector>
#include <map>

#include <tool/calc.h>

namespace classifier {
namespace pa {
class PA {
 public:
  explicit PA(size_t mode = 0);
  ~PA() {};

  void SetC(double C);

  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv,
            std::string* predict) const;
  void CompareFeatureWeight(const std::string& feature,
                            std::vector<std::pair<std::string, double> >* results) const;

 private:
  void CalcScores(const feature_vector& fv,
                  std::vector<std::pair<double, std::string> >* score2class) const;

  double CalcHingeLoss(const std::vector<std::pair<double, std::string> >& score2class,
                       const std::string& correct,
                       std::string* non_correct_predict) const;

  void Update(const std::string& correct,
              const std::vector<std::pair<double, std::string> >& score2class,
              const feature_vector& fv);

  weight_matrix weight_;
  size_t mode_;
  double C_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_PASSIVE_AGGRESSIVE_PA_H_
