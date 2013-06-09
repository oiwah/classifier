#include "nb.h"

#include <cmath>

namespace classifier {
namespace naivebayes {
NaiveBayes::NaiveBayes() : smoothing_(false), alpha_(0.0), document_sum_(0) {
  document_vector().swap(document_count_);
  word_sum_vector().swap(word_sum_in_each_category_);
  word_matrix().swap(word_count_in_each_category_);
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
  for (std::vector<datum>::const_iterator it = data.begin();
       it != data.end();
       ++it) {
    ++document_sum_;
    std::string category = it->category;
    
    CountCategory(category);
    CountWord(category, it->fv);
  }
}

void NaiveBayes::Test(const feature_vector& fv, std::string* result) const {
  *result = non_class;
  double score = non_class_score;

  for (word_matrix::const_iterator it = word_count_in_each_category_.begin();
       it != word_count_in_each_category_.end();
       ++it) {
    std::string category = it->first;
    double probability = CalculateProbability(fv, category);

    if (*result == non_class || score < probability) {
      *result = category;
      score = probability;
    }
  }
}

void NaiveBayes::GetFeatureWeight(size_t feature_id,
                                  std::vector<std::pair<std::string, double> >* results) const {
  for (word_matrix::const_iterator it = word_count_in_each_category_.begin();
       it != word_count_in_each_category_.end();
       ++it) {
    std::string category = it->first;
    if (it->second.size() <= feature_id) {
      results->push_back(make_pair(category, 0.0));
    } else {
      double score = it->second.at(feature_id) / word_sum_in_each_category_.at(category);
      results->push_back(make_pair(category, score));
    }
  }
}

void NaiveBayes::CountCategory(const std::string& category) {
  if (document_count_.find(category) == document_count_.end()) {
    document_count_[category] = 1;
    word_sum_in_each_category_[category] = 0.0;
    word_count_in_each_category_.insert(make_pair(category, word_vector()));
  } else {
    ++document_count_[category];
  }
}

void NaiveBayes::CountWord(const std::string& category,
                           const feature_vector& fv) {
  word_vector &word_count = word_count_in_each_category_[category];
  double &word_sum = word_sum_in_each_category_[category];
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    size_t word_id = it->first;
    double count = it->second;

    if (word_count.size() <= word_id)
      word_count.resize(word_id+1, 0.0);

    word_count[word_id] += count;
    word_sum += count;
  }
}

double NaiveBayes::CalculateProbability(const feature_vector& fv,
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
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    size_t word_id = it->first;
    const word_vector &word_count_in_a_category
        = word_count_in_each_category_.at(category);
    if (word_id < word_count_in_a_category.size()) {
      probability += log(
          (word_count_in_a_category.at(word_id) + smoothing_parameter)
          / (word_sum_in_each_category_.at(category) + (fv.size() * smoothing_parameter)) )
          * it->second;
    } else {
      if (!smoothing_) {
        probability = non_class_score;
        break;
      }

      // Approximate the number of word summation
      probability += log(
          smoothing_parameter /
          (word_sum_in_each_category_.at(category)
           + (fv.size() * smoothing_parameter)) )
           * it->second;
    }
  }

  //std::cout << category << " : " << probability << std::endl;
  return probability;
}

} //namespace
} //namespace
