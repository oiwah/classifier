#ifndef CLASSIFIER_CONFIDENCE_WEIGHTED_CW_H_
#define CLASSIFIER_CONFIDENCE_WEIGHTED_CW_H_

#include <iostream>
#include <vector>

#include <tool/calc.h>

namespace classifier {
namespace cw {
typedef std::unordered_map<std::string, double> covariance_vector;
typedef std::unordered_map<std::string, covariance_vector> covariance_matrix;

class CW {
 public:
  explicit CW(double phi = 0.0);
  ~CW() {};

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

  double CalcV(const feature_vector& fv,
               const std::string& correct,
               const std::string& non_correct_predict);

  double CalcAlpha(double m, double v) const;

  void Update(const std::string& correct,
              const score2class& scores,
              const feature_vector& fv);

  weight_matrix weight_;
  covariance_matrix cov_;
  double phi_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_CONFIDENCE_WEIGHTED_CW_H_
