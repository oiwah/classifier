#ifndef CLASSIFIER_NAIVEBAYES_H
#define CLASSIFIER_NAIVEBAYES_H

#include <iostream>
#include <vector>
#include <set>
#include <map>

namespace classifier {
namespace naivebayes {
struct datum {
  std::string category;
  std::vector<std::string> words;
};

class NaiveBayes {
 public:
  NaiveBayes();
  ~NaiveBayes() {};

  void set_alpha(double alpha);
  
  void Train(const std::vector<datum>& data);
  void Test(const datum& datum, std::string* result) const;
  void CompareFeatureWeight(const std::string& feature,
                            std::vector<std::pair<std::string, double> >* results) const;

 private:
  void CountWord(const std::string& category,
                 const std::vector<std::string>& words);

  //smoothing parameter
  bool smoothing_;
  double alpha_;
  
  size_t document_sum_;
  std::map<std::string, size_t> document_count_;
  
  std::map<std::string, size_t> word_sum_in_each_category_;
  std::map<std::string, std::map<std::string, size_t> > word_count_in_each_category_;
};

} //namespace
} //namespace

#endif //CLASSIFIER_NAIVEBAYES_H
