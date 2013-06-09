#ifndef CLASSIFIER_NAIVEBAYES_NB_H_
#define CLASSIFIER_NAIVEBAYES_NB_H_

#include <iostream>
#include <vector>
#include <unordered_map>

#include "../../utility/feature.h"

namespace classifier {
namespace naivebayes {
typedef std::unordered_map<std::string, size_t> document_vector;
typedef std::vector<double> word_vector;
typedef std::unordered_map<std::string, word_vector> word_matrix;
typedef std::unordered_map<std::string, double> word_sum_vector;

class NaiveBayes {
 public:
  NaiveBayes();
  virtual ~NaiveBayes() {};

  void set_alpha(double alpha);
  
  void Train(const std::vector<datum>& data);
  void Test(const feature_vector& fv, std::string* result) const;
  void GetFeatureWeight(size_t feature_id,
                        std::vector<std::pair<std::string, double> >* results) const;
 protected:
  bool smoothing_;
  double alpha_; //smoothing parameter

  size_t document_sum_;
  document_vector document_count_;

  word_sum_vector word_sum_in_each_category_;
  word_matrix word_count_in_each_category_;

 private:
  void CountCategory(const std::string& category);
  void CountWord(const std::string& category,
                 const feature_vector& fv);

  virtual double CalculateProbability(const feature_vector& fv,
                                      const std::string& category) const;
};

} //namespace
} //namespace

#endif //CLASSIFIER_NAIVEBAYES_NB_H_
