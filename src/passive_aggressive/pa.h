#ifndef CLASSIFIER_PASSIVE_AGGRESSIVE_PA_H_
#define CLASSIFIER_PASSIVE_AGGRESSIVE_PA_H_

#include <iostream>
#include <vector>

#include <tool/calc.h>

namespace classifier {
namespace pa {
class PA {
 public:
  explicit PA(size_t mode = 0);
  ~PA() {};

  void SetC(double C);

  void Train(const datum& datum);
  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv,
            std::string* predict) const;
  void GetFeatureWeight(const std::string& feature,
                        std::vector<std::pair<std::string, double> >* results) const;

 private:
  void CalcScores(const feature_vector& fv,
                  score2class* scores) const;

  void Update(const std::string& correct,
              const score2class& scores,
              const feature_vector& fv);

  weight_matrix weight_;
  size_t mode_;
  double C_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_PASSIVE_AGGRESSIVE_PA_H_
