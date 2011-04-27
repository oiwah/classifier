#include "nb.hpp"

namespace classifier {
namespace naivebayes {
NaiveBayes::NaiveBayes() : document_sum(0) {
  std::map<std::string, size_t>().swap(document_count);
  std::map<std::string, size_t>().swap(category_word_sum);
  std::map<std::string, std::map<std::string, size_t> >().swap(word_count);
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
    double probability = it->second / (double)document_sum;

    for (size_t i = 0; i < datum.words.size(); ++i) {
      if (word_count[it->first].find(datum.words[i]) == word_count[it->first].end()) {
        probability = -1;
        break;
      } else {
        probability *= word_count[it->first][datum.words[i]]
            / (double)category_word_sum[it->first];
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
