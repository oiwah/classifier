#ifndef CLASSIFIER_CONFIDENCE_WEIGHTED_SCW_H_
#define CLASSIFIER_CONFIDENCE_WEIGHTED_SCW_H_

#include <iostream>
#include <vector>

#include <tool/calc.h>

namespace classifier {
namespace scw {
typedef std::vector<double> covariance_vector;
typedef std::unordered_map<std::string, covariance_vector> covariance_matrix;

class SCW {
 public:
  explicit SCW(double phi = 0.0);
  ~SCW() {};

  void Train(const datum& datum);
  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv,
            std::string* predict) const;
  void GetFeatureWeight(size_t feature_id,
                        std::vector<std::pair<std::string, double> >* results) const;

  void SetC(double C) { C_ = C; }
  void ChangeMode(int mode) { mode_ = mode; }

 private:
  void CalcScores(const feature_vector& fv,
                  score2class* scores) const;

  double CalcV(const datum& datum,
               const std::string& non_correct_predict);

  double CalcAlpha(double m, double v) const;
  double CalcAlpha1(double m, double v) const;
  double CalcAlpha2(double m, double v) const;

  double CalcBeta(double v, double alpha) const;

  void Update(const datum& datum, const score2class& scores);

  int mode_;

  weight_matrix weight_;
  covariance_matrix cov_;
  double phi_;

  double C_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_CONFIDENCE_WEIGHTED_SCW_H_
