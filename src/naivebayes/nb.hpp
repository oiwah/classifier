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

  void Train(const std::vector<datum>& data);
  void Test(datum& datum);

 private:
  void CountWord(const std::string& category,
                 const std::vector<std::string>& words);

  size_t document_sum;
  std::map<std::string, size_t> document_count;
  
  std::map<std::string, size_t> category_word_sum;
  std::map<std::string, std::map<std::string, size_t> > word_count;
};

} //namespace
} //namespace

#endif //CLASSIFIER_NAIVEBAYES_H
