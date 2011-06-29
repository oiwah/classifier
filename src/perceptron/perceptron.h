#ifndef CLASSIFIER_PERCEPTRON_PERCEPTRON_H_
#define CLASSIFIER_PERCEPTRON_PERCEPTRON_H_

#include <iostream>
#include <vector>
#include <map>

#include <tool/feature.h>

namespace classifier {
namespace perceptron {
const std::string non_class = "None";

class Perceptron {
 public:
  Perceptron();
  ~Perceptron() {};

  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv, std::string* predict) const;
  void CompareFeatureWeight(const std::string& feature,
                            std::vector<std::pair<std::string, double> >* results) const;

 private:
  double InnerProduct(const feature_vector& fv,
                      weight_vector& wv) const;

  void Update(const feature_vector& fv,
              const std::string& correct,
              const std::string& predict);

  weight_matrix weight_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_PERCEPTRON_PERCEPTRON_H_
