#include <naivebayes/nb.h>

#include <cmath>

namespace classifier {
namespace naivebayes {
NaiveBayes::NaiveBayes() : smoothing_(false), alpha_(0.0), document_sum_(0) {
  std::map<std::string, size_t>().swap(document_count_);
  std::map<std::string, size_t>().swap(word_sum_in_each_category_);
  std::map<std::string, feature_vector>().swap(word_count_in_each_category_);
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
    
    CountCategory(category);
    CountWord(category, data[i].fv);
  }
}

void NaiveBayes::Test(const datum& datum, std::string* result) const {
  *result = non_class;
  double score = non_class_score;

  for (std::map<std::string, feature_vector>::const_iterator it =
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
  for (std::map<std::string, feature_vector>::const_iterator it =
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

void NaiveBayes::CountCategory(const std::string& category) {
  if (document_count_.find(category) == document_count_.end()) {
    document_count_[category] = 1;
    word_sum_in_each_category_[category] = 0;
    word_count_in_each_category_.insert(make_pair(category, feature_vector()));
  } else {
    ++document_count_[category];
  }
}

void NaiveBayes::CountWord(const std::string& category,
                           const feature_vector& fv) {
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    std::string word = it->first;
    double count = it->second;

    word_count_in_each_category_[category][word] += count;
    word_sum_in_each_category_[category] += count;
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
  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    std::string word = it->first;
    const feature_vector &word_count_in_a_category
        = word_count_in_each_category_.at(category);
    if (word_count_in_a_category.find(word) == word_count_in_a_category.end()) {
      if (!smoothing_) {
        probability = non_class_score;
        break;
      }

      // Approximate the number of word summation
      probability += log(
          smoothing_parameter /
          ((double)word_sum_in_each_category_.at(category)
           + (datum.fv.size() * smoothing_parameter)) )
           * it->second;
    } else {
      probability += log(
          (word_count_in_a_category.at(word) + smoothing_parameter)
          / ((double)word_sum_in_each_category_.at(category) + (datum.fv.size() * smoothing_parameter)) )
          * it->second;
    }
  }

  //std::cout << category << " : " << probability << std::endl;
  return probability;
}

} //namespace
} //namespace
