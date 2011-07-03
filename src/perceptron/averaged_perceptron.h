#ifndef CLASSIFIER_PERCEPTRON_AVERAGED_PERCEPTRON_H_
#define CLASSIFIER_PERCEPTRON_AVERAGED_PERCEPTRON_H_

#include <iostream>
#include <vector>
#include <map>

#include <tool/calc.h>

namespace classifier {
namespace perceptron {
class AveragedPerceptron {
 public:
  AveragedPerceptron();
  ~AveragedPerceptron() {};

  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv, std::string* predict) const;
  void CompareFeatureWeight(const std::string& feature,
                            std::vector<std::pair<std::string, double> >* results) const;

 private:
  void Update(const feature_vector& fv,
              const std::string& correct,
              const std::string& predict);

  void Predict(bool averaged,
               const feature_vector& fv,
               std::string* predict) const;

  void CalcAveragedWeight();
  weight_matrix weight_;
  weight_matrix differential_weight_;
  weight_matrix averaged_weight_;
  size_t dataN_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_PERCEPTRON_AVERAGED_PERCEPTRON_H_
