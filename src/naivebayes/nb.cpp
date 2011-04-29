#include "nb.hpp"

namespace classifier {
namespace naivebayes {
NaiveBayes::NaiveBayes() : smoothing(false), alpha(0.0), document_sum(0) {
  std::map<std::string, size_t>().swap(document_count);
  std::map<std::string, size_t>().swap(category_word_sum);
  std::map<std::string, std::map<std::string, size_t> >().swap(word_count);
}

void NaiveBayes::set_alpha(double alpha_) {
  if (alpha_ <= 1.0) {
    std::cerr << "you must set alpha more than 1.0" << std::endl;
  } else {
    if (!smoothing) smoothing = true;
    alpha = alpha_;
  }
}

void NaiveBayes::Train(const std::vector<datum>& data) {
  for (size_t i = 0; i < data.size(); ++i) {
    ++document_sum;
    std::string category = data[i].category;
    
    if (document_count.find(category) == document_count.end()) {
      document_count.insert(make_pair(category, 1));
      category_word_sum.insert(make_pair(category, 0));
      word_count.insert(make_pair(category, std::map<std::string, size_t>()));
    } else {
      ++document_count[category];
    }
    
    CountWord(category, data[i].words);
  }
}

void NaiveBayes::Test(datum& datum) {
  datum.category = "None";
  double score = -1;

  for (std::map<std::string, size_t>::iterator it = document_count.begin();
       it != document_count.end();
       ++it) {
    double probability = 1.0;
    double smoothing_parameter = 0.0;
    if (smoothing)
      smoothing_parameter = alpha - 1.0;

    // Class Probability
    probability *= (it->second + smoothing_parameter)
        / ((double)document_sum + document_count.size() * smoothing_parameter);

    // Word Probability
    for (size_t i = 0; i < datum.words.size(); ++i) {
      if (word_count[it->first].find(datum.words[i]) == word_count[it->first].end()) {
        if (!smoothing) {
          probability = -1;
          break;
        }
        
        // Approximate the number of word distribution
        probability *= smoothing_parameter /
            ((double)category_word_sum[it->first] + (datum.words.size()) * smoothing_parameter);
      } else {
        probability *= (word_count[it->first][datum.words[i]] + smoothing_parameter)
            / ((double)category_word_sum[it->first] + (datum.words.size()) * smoothing_parameter);
      }
    }

    if (score < probability) {
      datum.category = it->first;
      score = probability;
    }
  }
}

void NaiveBayes::CountWord(const std::string& category,
                           const std::vector<std::string>& words) {
  for (size_t i = 0; i < words.size(); ++i) {
    std::string word = words[i];
    if (word_count[category].find(word) == word_count[category].end()) {
      word_count[category].insert(make_pair(word, 1));
    } else {
      ++word_count[category][word];
    }
    ++category_word_sum[category];
  }
}

} //namespace
} //namespace
