#ifndef CLASSIFIER_FOBOS_FOBOS_H_
#define CLASSIFIER_FOBOS_FOBOS_H_

#include <iostream>
#include <vector>
#include <map>

#include <tool/calc.h>

namespace classifier {
namespace fobos {
class FOBOS {
 public:
  FOBOS(double eta = 1.0,
        double lambda = 1.0);
  ~FOBOS() {};

  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  void Test(const feature_vector& fv, std::string* predict) const;
  void GetFeatureWeight(const std::string& feature,
                        std::vector<std::pair<std::string, double> >* results) const;

 private:
  void Truncate(const std::string correct,
                const std::string non_correct_predict,
                const feature_vector& fv);

  void CalcScores(const feature_vector& fv,
                  std::vector<std::pair<double, std::string> >* score2class) const;

  void Update(const std::string& correct,
              const std::string& non_correct_predict,
              const double hinge_loss,
              const feature_vector& fv);

  double CalcHingeLoss(const datum& datum,
                       std::string* non_correct_predict) const;

  weight_matrix weight_;
  weight_matrix prev_updateN_;

  size_t dataN_;
  double eta_;
  double lambda_;
  double truncate_sum_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_FOBOS_FOBOS_H_
