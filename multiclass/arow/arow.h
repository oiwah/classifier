#ifndef CLASSIFIER_AROW_AROW_H_
#define CLASSIFIER_AROW_AROW_H_

#include <iostream>
#include <vector>

#include "../../utility/calc.h"

namespace classifier {
namespace arow {
typedef std::vector<double> covariance_vector;
typedef std::unordered_map<std::string, covariance_vector> covariance_matrix;

class AROW {
 public:
  explicit AROW(double phi = 0.0);
  ~AROW() {};

  void Train(const datum& datum);
  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv,
            std::string* predict) const;
  void GetFeatureWeight(size_t feature_id,
                        std::vector<std::pair<std::string, double> >* results) const;

 private:
  void CalcScores(const feature_vector& fv,
                  score2class* scores) const;

  double CalcV(const datum& datum,
               const std::string& non_correct_predict);

  void Update(const datum& datum,
              const score2class& scores);

  weight_matrix weight_;
  covariance_matrix cov_;
  double phi_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_AROW_AROW_H_
