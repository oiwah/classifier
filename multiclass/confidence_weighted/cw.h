#ifndef CLASSIFIER_CONFIDENCE_WEIGHTED_CW_H_
#define CLASSIFIER_CONFIDENCE_WEIGHTED_CW_H_

#include <iostream>
#include <vector>

#include "../../utility/calc.h"

/**
 *  CW (mode:0)
 *  Confidence-weighted linear classification.
 *  Dredze et al. ICML 2008
 *  http://www.cs.jhu.edu/~mdredze/publications/icml_variance.pdf
 */

/**
 *  SCW-I (mode:1), SCW-II (mode:2)
 *  Soft Confidence-Weighted Learning.
 *  Jialei et al. ICML2012
 *  http://icml.cc/2012/papers/86.pdf
 */

namespace classifier {
namespace cw {
typedef std::vector<double> covariance_vector;
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
  double CalcAlpha0(double m, double v) const;
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

#endif //CLASSIFIER_CONFIDENCE_WEIGHTED_CW_H_
