#include <complement_nb/complement_nb.h>

#include <cmath>
#include <cfloat>

namespace classifier {
namespace naivebayes {
double ComplementNaiveBayes::CalculateProbability(const datum& datum,
                                                  const std::string& category) const {
  double probability = 0.0;
  double smoothing_parameter = 0.0;
  if (smoothing_)
    smoothing_parameter = alpha_ - 1.0;

  // Class Probability
  probability -= log(
      ((document_sum_ - document_count_.at(category)) + smoothing_parameter) /
      ((double)document_sum_ + document_count_.size() * smoothing_parameter) );

  // Calculate Word Sum Except One Category
  size_t word_sum_except_a_category = 0.0;
  for (std::map<std::string, size_t>::const_iterator it =
           word_sum_in_each_category_.begin();
       it != word_sum_in_each_category_.end();
       ++it) {
    if (it->first == category) continue;
    word_sum_except_a_category += it->second;
  }

  // Calculate Word Count Except One Category
  for (size_t i = 0; i < datum.words.size(); ++i) {
    std::string word = datum.words[i];
    size_t word_count_except_a_category = 0;
    for (std::map<std::string, std::map<std::string, size_t> >::const_iterator it =
             word_count_in_each_category_.begin();
         it != word_count_in_each_category_.end();
         ++it) {
      if (it->first == category) continue;

      const std::map<std::string, size_t> &word_count_in_a_category
          = word_count_in_each_category_.at(it->first);
      if (word_count_in_a_category.find(word) == word_count_in_a_category.end())
        continue;
      else
        word_count_except_a_category += word_count_in_a_category.at(word);
    }

    // Word Probability
    if (word_count_except_a_category == 0) {
      if (!smoothing_) {
        probability = -DBL_MAX;
        break;
      }

      // Approximate the number of word summation
      probability -= log(
          smoothing_parameter /
          ((double)word_sum_except_a_category
           + (datum.words.size() * smoothing_parameter)) );
    } else {
        probability -= log(
            (word_count_except_a_category + smoothing_parameter)
            / ((double)word_sum_except_a_category
               + (datum.words.size() * smoothing_parameter)) );
    }
  }

  std::cout << category << " : " << probability << std::endl;
  return probability;
}

} //namespace
} //namespace
