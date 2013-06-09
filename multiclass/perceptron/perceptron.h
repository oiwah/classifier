#ifndef CLASSIFIER_PERCEPTRON_PERCEPTRON_H_
#define CLASSIFIER_PERCEPTRON_PERCEPTRON_H_

#include <vector>
#include <iostream>

#include "../../utility/calc.h"

namespace classifier {
namespace perceptron {
class Perceptron {
 public:
  Perceptron();
  ~Perceptron() {};

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
              const std::string& predict);

  weight_matrix weight_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_PERCEPTRON_PERCEPTRON_H_
