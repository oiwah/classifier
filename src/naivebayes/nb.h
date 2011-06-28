#ifndef CLASSIFIER_NAIVEBAYES_NB_H_
#define CLASSIFIER_NAIVEBAYES_NB_H_

#include <iostream>
#include <vector>
#include <set>
#include <map>

#include <tool/feature.h>

namespace classifier {
namespace naivebayes {

class NaiveBayes {
 public:
  NaiveBayes();
  virtual ~NaiveBayes() {};

  void set_alpha(double alpha);
  
  void Train(const std::vector<datum>& data);
  void Test(const datum& datum, std::string* result) const;
  void CompareFeatureWeight(const std::string& feature,
                            std::vector<std::pair<std::string, double> >* results) const;

 protected:
  bool smoothing_;
  double alpha_; //smoothing parameter

  size_t document_sum_;
  std::map<std::string, size_t> document_count_;

  std::map<std::string, size_t> word_sum_in_each_category_;
  std::map<std::string, feature_vector> word_count_in_each_category_;

 private:
  void CountCategory(const std::string& category);
  void CountWord(const std::string& category,
                 const feature_vector& fv);

  virtual double CalculateProbability(const datum& datum,
                                      const std::string& category) const;
};

} //namespace
} //namespace

#endif //CLASSIFIER_NAIVEBAYES_NB_H_
