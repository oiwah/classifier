#include <naivebayes/nb.h>

#include <cmath>
#include <cfloat>

namespace classifier {
namespace naivebayes {
NaiveBayes::NaiveBayes() : smoothing_(false), alpha_(0.0), document_sum_(0) {
  std::map<std::string, size_t>().swap(document_count_);
  std::map<std::string, size_t>().swap(word_sum_in_each_category_);
  std::map<std::string, std::map<std::string, size_t> >().swap(word_count_in_each_category_);
}

void NaiveBayes::set_alpha(double alpha) {
  if (alpha <= 1.0) {
    std::cerr << "you must set alpha more than 1.0" << std::endl;
  } else {
    if (!smoothing_) smoothing_ = true;
    alpha_ = alpha;
  }
}

void NaiveBayes::Train(const std::vector<datum>& data) {
  for (size_t i = 0; i < data.size(); ++i) {
    ++document_sum_;
    std::string category = data[i].category;
    
    if (document_count_.find(category) == document_count_.end()) {
      document_count_.insert(make_pair(category, 1));
      word_sum_in_each_category_.insert(make_pair(category, 0));
      word_count_in_each_category_.insert(
          make_pair(category, std::map<std::string, size_t>()));
    } else {
      ++document_count_[category];
    }

    CountWord(category, data[i].words);
  }
}

void NaiveBayes::Test(const datum& datum, std::string* result) const {
  *result = "None";
  double score = -DBL_MAX;

  for (std::map<std::string, std::map<std::string, size_t> >::const_iterator it =
           word_count_in_each_category_.begin();
       it != word_count_in_each_category_.end();
       ++it) {
    std::string category = it->first;
    double probability = CalculateProbability(datum, category);

    if (score < probability) {
      *result = category;
      score = probability;
    }
  }
}

void NaiveBayes::CompareFeatureWeight(const std::string& feature,
                                      std::vector<std::pair<std::string, double> >* results) const {
  for (std::map<std::string, std::map<std::string, size_t> >::const_iterator it =
           word_count_in_each_category_.begin();
       it != word_count_in_each_category_.end();
       ++it) {
    std::string category = it->first;
    if (it->second.find(feature) == it->second.end()) {
      results->push_back(make_pair(category, 0.0));
    } else {
      double score = it->second.at(feature) / (double)word_sum_in_each_category_.at(category);
      results->push_back(make_pair(category, score));
    }
  }
}


void NaiveBayes::CountWord(const std::string& category,
                           const std::vector<std::string>& words) {
  for (size_t i = 0; i < words.size(); ++i) {
    std::string word = words[i];
    if (word_count_in_each_category_[category].find(word)
        == word_count_in_each_category_[category].end()) {
      word_count_in_each_category_[category].insert(make_pair(word, 1));
    } else {
      ++word_count_in_each_category_[category][word];
    }
    ++word_sum_in_each_category_[category];
  }
}

double NaiveBayes::CalculateProbability(const datum& datum,
                                        const std::string& category) const {
  double probability = 0.0;
  double smoothing_parameter = 0.0;
  if (smoothing_)
    smoothing_parameter = alpha_ - 1.0;

  // Class Probability
  probability += log(
      (document_count_.at(category) + smoothing_parameter) /
      ((double)document_sum_ + document_count_.size() * smoothing_parameter) );

  // Word Probability
  for (size_t i = 0; i < datum.words.size(); ++i) {
    std::string word = datum.words[i];
    const std::map<std::string, size_t> &word_count_in_a_category
        = word_count_in_each_category_.at(category);
    if (word_count_in_a_category.find(word) == word_count_in_a_category.end()) {
      if (!smoothing_) {
        probability = -DBL_MAX;
        break;
      }

      // Approximate the number of word distribution
      probability += log(
          smoothing_parameter /
          ((double)word_sum_in_each_category_.at(category)
           + (datum.words.size() * smoothing_parameter)) );
    } else {
      probability += log(
          (word_count_in_a_category.at(word) + smoothing_parameter)
          / ((double)word_sum_in_each_category_.at(category)
             + (datum.words.size() * smoothing_parameter)) );
    }
  }

  std::cout << category << " : " << probability << std::endl;
  return probability;
}

} //namespace
} //namespace
